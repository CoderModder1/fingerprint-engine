from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.handlers.pdf_handler import PDFFileHandler


def test_pdf_handler_missing_pypdf_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A missing pypdf must fail loud with MissingDependencyError rather than be
    # silently swallowed into the latin1 byte fallback, which would produce a
    # divergent fingerprint incomparable to one made where pypdf is installed
    # (silent index corruption).
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\nsome real-looking pdf content\n%%EOF\n")

    # Drop any cached pypdf so the import inside load() re-runs, and force the
    # re-import to raise ImportError.
    monkeypatch.delitem(sys.modules, "pypdf", raising=False)
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(MissingDependencyError) as excinfo:
        PDFFileHandler().load(path)

    assert excinfo.value.package == "pypdf"
    assert excinfo.value.extra == "pdf"


def test_pdf_handler_corrupt_pdf_still_degrades_to_latin1(tmp_path: Path) -> None:
    # A genuine parse failure on a corrupt/unparseable PDF (pypdf installed)
    # must still degrade gracefully to the latin1 byte fallback rather than
    # raise -- only the missing-dependency case fails loud.
    pytest.importorskip("pypdf")

    path = tmp_path / "garbage.pdf"
    path.write_bytes(b"%PDF-1.4 garbage")

    payload = PDFFileHandler().load(path)

    assert payload.parser == "latin1_fallback"
    assert payload.page_count == 0
