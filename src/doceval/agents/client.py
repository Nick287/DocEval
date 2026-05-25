"""Construct an Agent Framework chat client wired up to Azure OpenAI.

Agent Framework's :class:`OpenAIChatClient` accepts an Azure endpoint and a
``credential`` parameter. When given an Azure ``TokenCredential`` the framework
derives an internal token provider and wires it as ``azure_ad_token_provider``
on the underlying ``AsyncAzureOpenAI`` client — which is the only AAD auth path
the OpenAI SDK recognises at request time. Using ``api_key=callable`` does NOT
reach that code path and fails with ``ValueError: Unable to handle auth``.

Note on ``api_version``:
    ``OpenAIChatClient`` internally calls the **Responses** API on the
    ``/openai/v1/responses`` (v1) URL path. That path only accepts the literal
    string ``"preview"`` (or ``"latest"``) as the ``api-version`` query value
    on Sweden Central — passing a dated value such as ``2025-04-01-preview``
    fails with ``BadRequest / API version not supported``.

    The plain ``AsyncAzureOpenAI`` client used by
    :mod:`doceval.sources.gpt_md_generator` hits the **legacy** Responses
    path ``/openai/responses`` instead and requires the opposite — a dated
    version like ``2025-04-01-preview`` (the literal ``"preview"`` returns
    ``404 Resource not found`` there).

    To keep both clients happy we leave the dated value in
    :class:`~doceval.config.Settings.azure_openai_api_version` for the
    legacy path, and pin the Responses v1 client to ``preview`` here.
"""
from __future__ import annotations

from functools import lru_cache

from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureCliCredential

from doceval.config import get_settings


@lru_cache(maxsize=1)
def _shared_credential() -> AzureCliCredential:
    # Pin to the tenant that owns the Azure OpenAI resource so we don't
    # accidentally use a token from another tenant (e.g. VS Code's signed-in
    # Microsoft corp identity).
    return AzureCliCredential(tenant_id=get_settings().azure_tenant_id)


@lru_cache(maxsize=1)
def build_chat_client() -> OpenAIChatClient:
    """Process-wide singleton chat client."""
    s = get_settings()
    return OpenAIChatClient(
        model=s.azure_openai_deployment,
        azure_endpoint=s.azure_openai_endpoint,
        # Responses v1 path on Sweden Central only accepts "preview" / "latest".
        # See module docstring for why we don't reuse s.azure_openai_api_version.
        api_version="preview",
        credential=_shared_credential(),
    )
