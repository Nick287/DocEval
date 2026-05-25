"""Vision verifier agent.

When a cluster has only one source backing it, we ask a vision-capable LLM
whether the candidate token is really present in the image. This is the only
place in the pipeline that calls an LLM, and it is intentionally narrow —
the agent has one job and returns structured JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_framework import Agent, Content, Message

from intsig_eval.agents.client import build_chat_client
from intsig_eval.config import get_settings


_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


_INSTRUCTIONS = (
    "You are a document verification assistant. You will be given an image of "
    "a document and a JSON list of candidate text strings (numbers, IDs, dates, "
    "etc.). For each candidate decide whether that exact string is visible in "
    "the document, reading the whole image carefully including margins, stamps, "
    "vertical text, table cells and small print.\n\n"
    "These candidates have already been declared suspect by other readers — "
    "they appear in only one source. Your job is to break the tie.\n\n"
    "Verdicts:\n"
    "  - present: the exact string (or an obvious visual match such as the "
    "same number with different separators, or the same date in a different "
    "format) is visible somewhere in the image.\n"
    "  - absent:  the string is NOT visible in the image.\n"
    "  - ambiguous: the region is unreadable or only partially matches.\n\n"
    "Return STRICT JSON of the form:\n"
    '  {"results": [{"token": "...", "verdict": "present|absent|ambiguous", '
    '"evidence": "用中文简述判断依据"}]}\n'
    "重要要求：evidence 字段必须使用中文。每个候选只返回一次，不要额外输出。"
)


def _image_content(image_path: Path) -> Content:
    suffix = image_path.suffix.lower()
    media = _MEDIA.get(suffix, "image/jpeg")
    return Content.from_data(image_path.read_bytes(), media_type=media)


class VisionVerifierAgent:
    """Thin wrapper around an :class:`agent_framework.Agent`.

    Exposes a single async method :meth:`verify` that maps a list of token
    surfaces to the agent's verdict for each. After each call,
    :attr:`last_model` carries the model string the service actually used
    (e.g. ``gpt-5.4-2025-09-xx``) — useful for auditing what version produced
    a given report. ``configured_model`` is the deployment alias requested at
    construction time.
    """

    def __init__(self, agent: Agent | None = None) -> None:
        s = get_settings()
        self._agent = agent or Agent(
            client=build_chat_client(),
            name="intsig-vision-verifier",
            instructions=_INSTRUCTIONS,
        )
        self.configured_model: str = s.azure_openai_deployment
        self.last_model: str | None = None

    async def verify(
        self,
        image_path: str | Path,
        candidates: list[str],
    ) -> dict[str, dict[str, str]]:
        """Verify whether each candidate appears in the image.

        Returns ``{surface: {"verdict": ..., "evidence": ...}}``.
        Surfaces missing from the response are absent from the returned dict.
        """
        if not candidates:
            return {}

        image_path = Path(image_path)
        user_text = (
            "Candidates JSON:\n"
            + json.dumps(candidates, ensure_ascii=False)
            + "\n\nReturn the verdict JSON now."
        )

        message = Message(
            role="user",
            contents=[Content.from_text(user_text), _image_content(image_path)],
        )

        response = await self._agent.run(message)
        self.last_model = self._extract_served_model(response) or self.last_model
        raw_text = self._extract_text(response)
        return self._parse(raw_text)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_served_model(response: object) -> str | None:
        """The agent framework stores the underlying ``ChatResponse`` on
        ``raw_representation``. Its ``model`` attribute is populated from the
        ``x-ms-served-model`` Azure response header (e.g. ``gpt-5.4-2025-...``).
        """
        raw = getattr(response, "raw_representation", None)
        for candidate in (raw, response):
            model = getattr(candidate, "model", None)
            if isinstance(model, str) and model:
                return model
        return None

    @staticmethod
    def _extract_text(response: object) -> str:
        """Pull the assistant's text out of an :class:`AgentResponse`."""
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            return text
        messages = getattr(response, "messages", None) or []
        chunks: list[str] = []
        for msg in messages:
            for c in getattr(msg, "contents", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    chunks.append(t)
        return "".join(chunks)

    @staticmethod
    def _parse(text: str) -> dict[str, dict[str, str]]:
        text = text.strip()
        if not text:
            return {}
        # Be lenient — strip ```json fences if the model adds them.
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to recover the first {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            else:
                return {}
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return {}
        out: dict[str, dict[str, str]] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token", "")).strip()
            verdict = str(item.get("verdict", "")).strip().lower()
            evidence = str(item.get("evidence", "")).strip()
            if token and verdict in {"present", "absent", "ambiguous"}:
                out[token] = {"verdict": verdict, "evidence": evidence}
        return out
