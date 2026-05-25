"""GPT vision → Markdown transcription generator.

Ports the single-image conversion logic from ``annot_test/image_to_markdown.py``
into the :mod:`doceval` package so the batch workflow can synthesize a
missing ``MD/gpt/<stem>.md`` on demand.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
from PIL import Image

from doceval.config import get_settings

log = logging.getLogger("doceval.gpt_md")


_SYSTEM_PROMPT = (
	"You are an enterprise-grade document parsing engine. Your task is to convert the "
	"user-provided document image into clean, faithful Markdown.\n"
	"\n"
	"Strictly follow these rules:\n"
	"1. Faithful extraction: Transcribe every piece of textual content visible in the image "
	"exactly as it appears, including headings, paragraphs, lists, form fields, labels, "
	"values, footnotes, and page numbers. Do not paraphrase, translate, summarize, or "
	"reorder content. Preserve the original language, casing, punctuation, numbers, and units.\n"
	"2. Reading order: Follow the natural top-to-bottom, left-to-right reading order of the "
	"document. For multi-column layouts, finish one column before moving to the next.\n"
	"3. Structure: Use Markdown headings (#, ##, ###) to reflect the visual hierarchy of titles "
	"and section headers. Use ordered or unordered lists where the source uses them. Use **bold** "
	"or *italic* only when the source clearly emphasizes the text.\n"
	"4. Tables: Render every table using standard Markdown table syntax with a header row and "
	"separator line. Preserve all rows and columns, including empty cells (use an empty string). "
	"Do not merge or split cells. If a cell spans multiple rows or columns in the source, repeat "
	"the value in each affected cell. Keep numeric values, currency symbols, and units exactly as shown.\n"
	"5. Key-value fields: For form-style fields (e.g. \"Invoice No: 12345\"), output them as "
	"`Label: Value` on a single line, or place them inside a two-column Markdown table when grouped together.\n"
	"6. Non-text elements: For barcodes, QR codes, signatures, stamps, logos, checkboxes, and other "
	"graphical elements, insert an inline tag in the form `<figure>short description, including any visible text or state</figure>` "
	"at the position where the element appears. For checkboxes, indicate the state, e.g. `<figure>checkbox: checked</figure>` or "
	"`<figure>checkbox: unchecked</figure>`.\n"
	"7. Illegible content: If a character or value is unreadable, use `[illegible]` in its place. Never invent or guess missing content.\n"
	"8. Noise filtering: Ignore purely decorative page borders, background watermarks, and scanner artifacts that carry no information.\n"
	"9. Output format: Return pure Markdown body content only. Do not include any preface, explanation, closing remarks, or code fences such as ```markdown."
)

_USER_TEXT_WITH_ROTATED = (
	"You will receive two views of the SAME document page:\n"
	"  - Image 1: the page in its original orientation. This is the primary view; transcribe it in full following the rules above.\n"
	"  - Image 2: the same page ROTATED 90° clockwise. Use it ONLY to read text that is printed vertically / sideways in the original (margin stamps, document IDs, tracking numbers along the edge, vertical watermarks, rotated cell labels in tables). Any extra text that becomes readable in Image 2 MUST also appear in your Markdown output, inserted at the position where it occurs in Image 1.\n\n"
	"Do not transcribe Image 2 separately and do not output content twice. Produce a SINGLE Markdown extraction of the page as seen in Image 1, enriched with the vertical/rotated text revealed by Image 2."
)

_MEDIA_MAP = {
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".png": "image/png",
	".gif": "image/gif",
	".webp": "image/webp",
}


@lru_cache(maxsize=1)
def _shared_credential() -> AzureCliCredential:
	# Pin to the tenant that owns the Azure OpenAI resource — see
	# :mod:`doceval.agents.client` for the full rationale.
	return AzureCliCredential(tenant_id=get_settings().azure_tenant_id)


@lru_cache(maxsize=1)
def _async_client() -> AsyncAzureOpenAI:
	s = get_settings()
	token_provider = get_bearer_token_provider(
		_shared_credential(),
		"https://cognitiveservices.azure.com/.default",
	)
	return AsyncAzureOpenAI(
		azure_ad_token_provider=token_provider,
		api_version=s.azure_openai_api_version,
		azure_endpoint=s.azure_openai_endpoint,
	)


def _media_type(image_path: Path) -> str:
	return _MEDIA_MAP.get(image_path.suffix.lower(), "image/jpeg")


def _image_to_base64(image_path: Path) -> str:
	return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _rotated_image_base64(image_path: Path) -> tuple[str, str]:
	"""Return ``(media_type, base64)`` for a 90°-CW rotated companion view."""
	with Image.open(image_path) as im:
		rotated = im.rotate(-90, expand=True)
		ext = image_path.suffix.lower()
		if ext in (".jpg", ".jpeg"):
			fmt, media = "JPEG", "image/jpeg"
			if rotated.mode not in ("RGB", "L"):
				rotated = rotated.convert("RGB")
		else:
			fmt, media = "PNG", "image/png"
		buf = io.BytesIO()
		save_kwargs: dict[str, Any] = {"quality": 92} if fmt == "JPEG" else {}
		rotated.save(buf, format=fmt, **save_kwargs)
		return media, base64.b64encode(buf.getvalue()).decode("utf-8")


async def generate_gpt_markdown(image_path: Path) -> str:
	"""Call Azure OpenAI to transcribe ``image_path`` to Markdown."""
	if not image_path.exists():
		raise FileNotFoundError(image_path)

	media_type = _media_type(image_path)
	b64 = await asyncio.to_thread(_image_to_base64, image_path)
	rot_media, rot_b64 = await asyncio.to_thread(_rotated_image_base64, image_path)

	client = _async_client()
	s = get_settings()

	log.info("gpt_md · request START  %s", image_path.name)
	response = await client.responses.create(
		model=s.azure_openai_deployment,
		instructions=_SYSTEM_PROMPT,
		input=[
			{
				"role": "user",
				"content": [
					{"type": "input_text", "text": _USER_TEXT_WITH_ROTATED},
					{"type": "input_image", "image_url": f"data:{media_type};base64,{b64}"},
					{"type": "input_image", "image_url": f"data:{rot_media};base64,{rot_b64}"},
				],
			},
		],
		max_output_tokens=16384,
	)
	content = response.output_text or ""
	log.info("gpt_md · request DONE   %s  (%d chars)", image_path.name, len(content))
	return content


async def ensure_gpt_markdown(
	image_path: Path, out_path: Path, *, overwrite: bool = False
) -> bool:
	"""Write ``MD/gpt/<stem>.md`` if missing. Return True if a call was made."""
	if out_path.exists() and not overwrite:
		return False
	content = await generate_gpt_markdown(image_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(content, encoding="utf-8")
	return True
