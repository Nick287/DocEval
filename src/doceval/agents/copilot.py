"""GitHub Copilot subscription client.

Lets the doceval pipeline call vision-capable models (``gpt-5.x``,
``claude-…``, ``gemini-…``) through a personal Copilot subscription instead
of an Azure OpenAI deployment.

⚠️  Using a Copilot subscription outside an IDE editing flow violates
    GitHub's Copilot terms of service and may get the account suspended.
    Provided here for **personal** experimentation only — do not use in
    production or for anything user-facing.

This module is intentionally self-contained. It mirrors the logic in
``Temp/git_model.py`` (see that file for design notes) and exposes:

* :func:`get_oauth_token` — read the long-lived ``ghu_…`` OAuth token from
  ``COPILOT_OAUTH_TOKEN`` / ``.env``, or trigger a Device Flow login (which
  writes the token back to ``.env``) if missing.
* :func:`get_api_token_async` — exchange the OAuth token for a short-lived
  (~30 min) API token + endpoint, cached in memory.
* :func:`responses_call` — POST a request to the Copilot ``/responses``
  endpoint (OpenAI Responses API shape).
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx


# Public client_id used by the official ``copilot.vim`` integration.
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_API_BASE = "https://api.githubcopilot.com"

# Mandatory editor-spoof headers — the Copilot backend rejects requests
# without them (HTTP 401 / 403).
EDITOR_HEADERS: dict[str, str] = {
    "Editor-Version": "vscode/1.95.0",
    "Editor-Plugin-Version": "copilot-chat/0.22.0",
    "Copilot-Integration-Id": "vscode-chat",
    "User-Agent": "GitHubCopilotChat/0.22.0",
}

# Short-lived API token cache. Refreshed automatically when within 5 min of
# expiry. Process-local; not persisted to disk.
_api_token_mem: dict[str, Any] = {}
_api_token_lock = asyncio.Lock()


# --------------------------------------------------------------------------- #
# .env discovery / write-back (used only for device-flow first-time setup)
# --------------------------------------------------------------------------- #
def _find_dotenv() -> Path | None:
    """Walk up from ``cwd`` looking for the first ``.env`` file."""
    cur = Path.cwd().resolve()
    for d in [cur, *cur.parents]:
        p = d / ".env"
        if p.is_file():
            return p
    return None


def _upsert_env_line(path: Path, key: str, value: str) -> None:
    """Add or update ``KEY=VALUE`` in ``path``, preserving other lines."""
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    new_line = f"{key}={value}"
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        pass


# --------------------------------------------------------------------------- #
# Device Flow → long-lived OAuth token
# --------------------------------------------------------------------------- #
def _device_flow_login() -> str:
    """Interactively obtain an OAuth token and persist it to ``.env``."""
    r = httpx.post(
        "https://github.com/login/device/code",
        headers={"Accept": "application/json"},
        data={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
        timeout=30,
    )
    r.raise_for_status()
    info = r.json()

    print("\n=== GitHub Copilot 登录 ===")
    print(f"请在浏览器打开: {info['verification_uri']}")
    print(f"输入此 code:     {info['user_code']}")
    print("授权后回到这里，等待自动检测...\n")

    interval = info.get("interval", 5)
    expires_at = time.time() + info.get("expires_in", 900)
    device_code = info["device_code"]

    while time.time() < expires_at:
        time.sleep(interval)
        r = httpx.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=30,
        )
        data = r.json()
        if "access_token" in data:
            token = data["access_token"]
            target = _find_dotenv() or (Path.cwd() / ".env")
            _upsert_env_line(target, "COPILOT_OAUTH_TOKEN", token)
            os.environ["COPILOT_OAUTH_TOKEN"] = token
            print(f"✅ 登录成功，token 已写入 {target}")
            return token
        if data.get("error") == "authorization_pending":
            continue
        if data.get("error") == "slow_down":
            interval += 5
            continue
        raise RuntimeError(f"Device flow 失败: {data}")

    raise TimeoutError("Device flow 超时，请重试")


def get_oauth_token() -> str:
    """Return the long-lived Copilot OAuth token.

    Resolution order:
      1. ``COPILOT_OAUTH_TOKEN`` env var (also populated from ``.env``).
      2. ``doceval.config.Settings.copilot_oauth_token``.
      3. Interactive device-flow login (blocks the calling thread).
    """
    token = os.environ.get("COPILOT_OAUTH_TOKEN")
    if token:
        return token
    # Late import to avoid a config ↔ agents circular dependency.
    from doceval.config import get_settings

    token = get_settings().copilot_oauth_token
    if token:
        return token
    return _device_flow_login()


# --------------------------------------------------------------------------- #
# Short-lived API token (≈30 min) — cached in memory
# --------------------------------------------------------------------------- #
def _exchange_for_api_token() -> tuple[str, str, float]:
    oauth = get_oauth_token()
    r = httpx.get(
        "https://api.github.com/copilot_internal/v2/token",
        headers={
            "Authorization": f"token {oauth}",
            "Accept": "application/json",
            **EDITOR_HEADERS,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"换 API token 失败 {r.status_code}: {r.text}\n"
            "可能原因：账号没有 Copilot 订阅，或 OAuth token 已失效"
            "（删除 .env 里的 COPILOT_OAUTH_TOKEN 重试）"
        )
    data = r.json()
    endpoint = data.get("endpoints", {}).get("api", COPILOT_API_BASE)
    return data["token"], endpoint, float(data["expires_at"])


async def get_api_token_async() -> tuple[str, str]:
    """Return ``(api_token, endpoint)``, refreshing the cache if needed."""
    now = time.time()
    if _api_token_mem and _api_token_mem.get("expires_at", 0) - 300 > now:
        return _api_token_mem["token"], _api_token_mem["endpoint"]

    async with _api_token_lock:
        # Re-check under the lock — another coroutine may have refreshed.
        if _api_token_mem and _api_token_mem.get("expires_at", 0) - 300 > time.time():
            return _api_token_mem["token"], _api_token_mem["endpoint"]
        token, endpoint, expires_at = await asyncio.to_thread(_exchange_for_api_token)
        _api_token_mem.update(token=token, endpoint=endpoint, expires_at=expires_at)
        return token, endpoint


# --------------------------------------------------------------------------- #
# /responses HTTP call (OpenAI Responses API shape)
# --------------------------------------------------------------------------- #
# Shared async HTTP client. ``openai`` already pulls in httpx so no new dep.
_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        # 600 s read timeout: large verifier batches (100+ candidates) routinely
        # take 3–5 min for GPT-5.5 to think through. Keep connect timeout tight.
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=15.0),
        )
    return _http_client


async def responses_call(payload: dict[str, Any]) -> dict[str, Any]:
    """POST ``payload`` to ``{endpoint}/responses`` and return parsed JSON.

    The payload must already be in OpenAI Responses API shape
    (``model``, ``input``, optional ``instructions``, etc.). Raises
    :class:`RuntimeError` on non-2xx.
    """
    token, endpoint = await get_api_token_async()
    r = await _client().post(
        f"{endpoint}/responses",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **EDITOR_HEADERS,
        },
        json=payload,
    )
    if r.status_code != 200:
        raise RuntimeError(f"copilot /responses {r.status_code}: {r.text}")
    return r.json()


def extract_response_text(data: dict[str, Any]) -> str:
    """Pull plain text out of a Copilot ``/responses`` JSON payload."""
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text"):
                chunks.append(c.get("text", ""))
    return "".join(chunks)


def extract_served_model(data: dict[str, Any]) -> str | None:
    """Return the model the Copilot backend actually routed to, if reported."""
    served = data.get("model")
    return served if isinstance(served, str) and served else None
