"""Handler for PDF text and lightweight structure extraction."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from .base import FileHandler
from .text_handler import TextFileHandler


@dataclass(frozen=True)
class PDFPayload:
    text: str
    page_count: int
    parser: str


class PDFFileHandler(FileHandler):
    name = "pdf"
    priority = 65
    default_signal_window = 512
    default_signal_hop = 128
    supported_mime_types = {"application/pdf"}
    supported_extensions = {".pdf"}

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        base_score = super().can_handle(path, mime_type, sample)
        if base_score:
            return base_score + 0.10
        if sample and sample.startswith(b"%PDF-"):
            return 0.95
        return 0.0

    def load(self, path: str | Path) -> PDFPayload:
        data = self.read_bytes(path)
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
            chunks: list[str] = []
            for page_index, page in enumerate(reader.pages):
                chunks.append(f"\n[[PAGE:{page_index + 1}]]\n")
                chunks.append(page.extract_text() or "")
            return PDFPayload(
                text="".join(chunks),
                page_count=len(reader.pages),
                parser="pypdf",
            )
        except Exception:
            text = data.decode("latin-1", errors="ignore")
            return PDFPayload(text=text, page_count=0, parser="latin1_fallback")

    def to_signal(self, payload: PDFPayload) -> np.ndarray:
        return TextFileHandler().to_signal(payload.text)

    def metadata(self, payload: PDFPayload) -> dict[str, object]:
        return {
            "page_count": payload.page_count,
            "parser": payload.parser,
            "character_count": len(payload.text),
            "signal_strategy": "pdf_text_with_page_markers",
        }
