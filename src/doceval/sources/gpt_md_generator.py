"""GPT vision → Markdown transcription generator.

Ports the single-image conversion logic from ``annot_test/image_to_markdown.py``
into the :mod:`doceval` package so the batch workflow can synthesize a
missing ``MD/<model_name>/<stem>.md`` on demand (e.g. ``MD/gpt-5.4/...``).

The actual model call goes through :func:`doceval.agents.llm.vision_responses`
so it transparently uses either Azure OpenAI or a GitHub Copilot subscription
depending on ``DOCEVAL_MODEL_SOURCE``.

Both the primary and 90°-rotated companion images are downscaled to
``Settings.vision_max_dim`` on the long side and re-encoded as JPEG before
being base64-d into the request — high-DPI scans easily push raw bytes past
Copilot's request-body cap and trigger HTTP 413. On 413 we retry once with
the long side halved.
"""
from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

from PIL import Image

from doceval.agents.llm import VisionImage, vision_responses
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

def _encode_view(
	im: Image.Image,
	*,
	max_dim: int,
	quality: int,
) -> VisionImage:
	"""Downscale (if needed) and JPEG-encode one image view."""
	if im.mode not in ("RGB", "L"):
		im = im.convert("RGB")
	w, h = im.size
	longest = max(w, h)
	if longest > max_dim:
		scale = max_dim / longest
		im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
	buf = io.BytesIO()
	im.save(buf, format="JPEG", quality=quality, optimize=True)
	return VisionImage(media_type="image/jpeg", data=buf.getvalue())


def _build_views(image_path: Path, *, max_dim: int, quality: int) -> tuple[VisionImage, VisionImage]:
	"""Return ``(primary, rotated_90cw)`` views ready for vision_responses."""
	with Image.open(image_path) as im:
		im.load()  # detach from file handle so we can rotate after closing
		primary = _encode_view(im, max_dim=max_dim, quality=quality)
		rotated_pil = im.rotate(-90, expand=True)
	rotated = _encode_view(rotated_pil, max_dim=max_dim, quality=quality)
	return primary, rotated


def _is_payload_too_large(exc: BaseException) -> bool:
	msg = str(exc)
	return "413" in msg or "too large" in msg.lower() or "payload" in msg.lower() and "size" in msg.lower()


async def generate_gpt_markdown(image_path: Path) -> str:
	"""Transcribe ``image_path`` to Markdown via the configured backend.

	Automatically retries once with a smaller ``max_dim`` if the backend
	rejects the request with HTTP 413 (payload too large).
	"""
	if not image_path.exists():
		raise FileNotFoundError(image_path)

	s = get_settings()
	# Try the configured size first; if the server says 413, shrink and retry.
	attempts: list[int] = [s.vision_max_dim, max(640, s.vision_max_dim // 2)]
	if attempts[0] == attempts[1]:
		attempts = attempts[:1]

	last_exc: BaseException | None = None
	for attempt_idx, max_dim in enumerate(attempts):
		primary, rotated = await asyncio.to_thread(
			_build_views,
			image_path,
			max_dim=max_dim,
			quality=s.vision_jpeg_quality,
		)
		payload_kb = (len(primary.data) + len(rotated.data)) // 1024
		log.info(
			"gpt_md · request START  %s  max_dim=%d  payload=%dKB  attempt=%d",
			image_path.name,
			max_dim,
			payload_kb,
			attempt_idx + 1,
		)
		try:
			result = await vision_responses(
				instructions=_SYSTEM_PROMPT,
				user_text=_USER_TEXT_WITH_ROTATED,
				images=[primary, rotated],
				max_output_tokens=16384,
			)
		except Exception as exc:
			last_exc = exc
			if _is_payload_too_large(exc) and attempt_idx + 1 < len(attempts):
				log.warning(
					"gpt_md · 413 payload-too-large at max_dim=%d (%dKB); retrying smaller",
					max_dim,
					payload_kb,
				)
				continue
			raise
		content = result.text
		log.info(
			"gpt_md · request DONE   %s  (%d chars, served=%s)",
			image_path.name,
			len(content),
			result.served_model or "-",
		)
		return content

	# Should be unreachable (loop either returns or re-raises), but keep
	# mypy / pyright happy and make the failure mode explicit.
	assert last_exc is not None
	raise last_exc


async def ensure_gpt_markdown(
	image_path: Path, out_path: Path, *, overwrite: bool = False
) -> bool:
	"""Write ``MD/<model_name>/<stem>.md`` if missing. Return True if a call was made."""
	if out_path.exists() and not overwrite:
		return False
	content = await generate_gpt_markdown(image_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(content, encoding="utf-8")
	return True
