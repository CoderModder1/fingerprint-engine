"""Handler for PDF text and lightweight structure extraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from fingerprint_engine.core.models import FingerprintConfig

from .base import FileHandler, require_optional
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

    def __init__(self, max_pdf_pages: int | None = None) -> None:
        # Page cap for untrusted PDFs (see SECURITY.md / FingerprintConfig).
        # ``None`` falls back to the config default (0 = unlimited), so the
        # no-arg construction the Fingerprinter uses keeps current behavior
        # while a caller (or a future config wiring) can bound page extraction.
        if max_pdf_pages is None:
            max_pdf_pages = FingerprintConfig().max_pdf_pages
        if max_pdf_pages < 0:
            raise ValueError("max_pdf_pages must be non-negative (0 = unlimited)")
        self.max_pdf_pages = max_pdf_pages

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

    def load(self, path: str | Path, *, content: bytes | None = None) -> PDFPayload:
        data = self.read_content(path, content)
        PdfReader = require_optional(
            "pypdf",
            package="pypdf",
            extra="pdf",
            message=(
                "pypdf is required to fingerprint PDF files; install it with "
                "'pip install \"fingerprint-engine[pdf]\"'"
            ),
        ).PdfReader

        try:
            reader = PdfReader(BytesIO(data))
            chunks: list[str] = []
            extracted_pages = 0
            for page_index, page in enumerate(reader.pages):
                # Bound work/memory on untrusted PDFs: when max_pdf_pages > 0,
                # stop after that many pages (0 = unlimited). Capping keeps a
                # multi-thousand-page document from blowing up extraction.
                if self.max_pdf_pages and extracted_pages >= self.max_pdf_pages:
                    break
                chunks.append(f"\n[[PAGE:{page_index + 1}]]\n")
                chunks.append(page.extract_text() or "")
                extracted_pages += 1
            return PDFPayload(
                text="".join(chunks),
                page_count=extracted_pages,
                parser="pypdf",
            )
        except Exception:
            # Genuine pypdf parse failure (corrupt/encrypted): degrade to the
            # latin1 byte fallback rather than raise. Warn loudly that this is a
            # STRUCTURAL-ONLY fingerprint of the raw bytes -- not real text
            # extraction -- so it is not silently mistaken for parsed content.
            warnings.warn(
                f"{Path(path).name}: pypdf could not parse this PDF; falling back "
                "to a structural-only fingerprint of the raw bytes (no text was "
                "extracted), which is not comparable to a text-extracted one.",
                RuntimeWarning,
                stacklevel=2,
            )
            text = data.decode("latin-1", errors="ignore")
            return PDFPayload(text=text, page_count=0, parser="latin1_fallback")

    def configure(self, config: FingerprintConfig) -> None:
        # Honor the configured page cap (0 = unlimited) so the page limit set
        # via FingerprintConfig / --max-pdf-pages reaches the handler that the
        # Fingerprinter discovers and instantiates with no arguments.
        self.max_pdf_pages = config.max_pdf_pages

    def to_signal(self, payload: PDFPayload) -> np.ndarray:
        return TextFileHandler().to_signal(payload.text)

    def metadata(self, payload: PDFPayload) -> dict[str, object]:
        return {
            "page_count": payload.page_count,
            "parser": payload.parser,
            "character_count": len(payload.text),
            "signal_strategy": "pdf_text_with_page_markers",
        }
