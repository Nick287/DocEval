from doceval.sources.base import TokenReader, discover_stems
from doceval.sources.markdown import MarkdownReader
from doceval.sources.ocr import AzureLayoutOCRReader

__all__ = [
    "AzureLayoutOCRReader",
    "MarkdownReader",
    "TokenReader",
    "discover_stems",
]
