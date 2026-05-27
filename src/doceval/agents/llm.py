"""Unified vision-capable LLM dispatch.

Both the singleton verifier (:mod:`doceval.agents.verifier`) and the
on-demand markdown generator (:mod:`doceval.sources.gpt_md_generator`) need
the same primitive: *given a system prompt, a user-text turn and one or
more inline images, return the assistant's text reply (plus the model id
the backend actually routed to)*.

This module exposes that primitive as :func:`vision_responses` and routes
to either Azure OpenAI or a personal GitHub Copilot subscription based on
``Settings.model_source``.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI

from doceval.agents import copilot
from doceval.config import get_settings


@dataclass(frozen=True)
class VisionImage:
    """One inline image attachment for a vision call."""

    media_type: str  # e.g. "image/jpeg"
    data: bytes      # raw image bytes (NOT base64 — encoded here)

    def to_data_url(self) -> str:
        b64 = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.media_type};base64,{b64}"


@dataclass(frozen=True)
class VisionResult:
    text: str
    served_model: str | None


# --------------------------------------------------------------------------- #
# Azure OpenAI path
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _aoai_credential() -> AzureCliCredential:
    # Pin to the tenant that owns the Azure OpenAI resource so we don't
    # accidentally use a token from another tenant.
    return AzureCliCredential(tenant_id=get_settings().azure_tenant_id)


@lru_cache(maxsize=1)
def _aoai_client() -> AsyncAzureOpenAI:
    s = get_settings()
    token_provider = get_bearer_token_provider(
        _aoai_credential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AsyncAzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version=s.azure_openai_api_version,
        azure_endpoint=s.azure_openai_endpoint,
    )


async def _aoai_vision_call(
    *,
    instructions: str,
    user_text: str,
    images: list[VisionImage],
    max_output_tokens: int | None,
    model: str,
) -> VisionResult:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    for img in images:
        content.append({"type": "input_image", "image_url": img.to_data_url()})

    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    response = await _aoai_client().responses.create(**kwargs)
    text = response.output_text or ""
    served = getattr(response, "model", None) or model
    return VisionResult(text=text, served_model=served)


# --------------------------------------------------------------------------- #
# GitHub Copilot path
# --------------------------------------------------------------------------- #
async def _copilot_vision_call(
    *,
    instructions: str,
    user_text: str,
    images: list[VisionImage],
    max_output_tokens: int | None,
    model: str,
) -> VisionResult:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    for img in images:
        content.append({"type": "input_image", "image_url": img.to_data_url()})

    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    data = await copilot.responses_call(payload)
    text = copilot.extract_response_text(data)
    served = copilot.extract_served_model(data) or model
    return VisionResult(text=text, served_model=served)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
async def vision_responses(
    *,
    instructions: str,
    user_text: str,
    images: list[VisionImage],
    max_output_tokens: int | None = None,
    model: str | None = None,
) -> VisionResult:
    """Send a vision prompt to the configured backend and return its reply.

    Dispatches on ``Settings.model_source``:

    * ``aoai``    → Azure OpenAI Responses API (legacy ``/openai/responses``).
    * ``copilot`` → GitHub Copilot ``/responses`` proxy.

    ``model`` overrides ``Settings.model_name`` for this single call — used
    by the verifier when ``DOCEVAL_VERIFIER_MODEL`` is set to something
    different from the generator model.
    """
    s = get_settings()
    resolved_model = model or s.model_name
    source = s.model_source
    if source == "copilot":
        return await _copilot_vision_call(
            instructions=instructions,
            user_text=user_text,
            images=images,
            max_output_tokens=max_output_tokens,
            model=resolved_model,
        )
    if source == "aoai":
        return await _aoai_vision_call(
            instructions=instructions,
            user_text=user_text,
            images=images,
            max_output_tokens=max_output_tokens,
            model=resolved_model,
        )
    raise ValueError(f"Unknown DOCEVAL_MODEL_SOURCE={source!r} (expected aoai|copilot)")
