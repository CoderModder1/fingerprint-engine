"""The Fingerprinter must propagate config-derived limits onto discovered handlers.

Handlers are auto-discovered and instantiated with no constructor arguments, so a
config value like the PDF page cap only takes effect if Fingerprinter pushes it
onto each handler via ``configure()``. Without that wiring the ``--max-pdf-pages``
flag / ``FingerprintConfig.max_pdf_pages`` is a silent no-op.
"""

from __future__ import annotations

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.models import FingerprintConfig


def test_configured_pdf_page_cap_reaches_discovered_handler() -> None:
    fingerprinter = Fingerprinter(FingerprintConfig(max_pdf_pages=3))
    pdf = next(handler for handler in fingerprinter.handlers if handler.name == "pdf")
    assert pdf.max_pdf_pages == 3


def test_default_pdf_page_cap_is_unlimited() -> None:
    fingerprinter = Fingerprinter()
    pdf = next(handler for handler in fingerprinter.handlers if handler.name == "pdf")
    assert pdf.max_pdf_pages == 0
