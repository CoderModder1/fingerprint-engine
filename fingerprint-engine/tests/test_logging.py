from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import FingerprintConfig

_TEXT_CONFIG = FingerprintConfig(
    window_size=32,
    hop_size=8,
    peak_threshold=0.25,
    peak_percentile=70.0,
    max_delta_t=16,
)


def test_handler_failure_emits_debug_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A swallowed per-handler decode failure (falling through to the next
    # candidate) must emit a DEBUG record on the package logger so routing
    # fallbacks are observable -- without changing the fall-through behavior.
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096)

    fingerprinter = Fingerprinter(FingerprintConfig())
    image_handler = next(h for h in fingerprinter.handlers if h.name == "image")

    def _raise_decode(_path: Path, *, content: bytes | None = None) -> object:
        raise ValueError("cannot decode image content")

    monkeypatch.setattr(image_handler, "load", _raise_decode)

    with caplog.at_level(logging.DEBUG, logger="fingerprint_engine"):
        fingerprint = fingerprinter.fingerprint_file(path)

    # Behavior unchanged: the image handler's failure demotes to binary.
    assert fingerprint.handler == "binary"

    debug_records = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG and record.name == "fingerprint_engine.core.fingerprinter"
    ]
    expected = f"handler image failed for {path}: cannot decode image content"
    assert any(
        record.getMessage() == expected for record in debug_records
    ), [record.getMessage() for record in debug_records]


def test_missing_dependency_emits_warning_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A re-raised MissingDependencyError must emit a WARNING naming the package
    # so a misconfigured environment is observable in logs.
    from fingerprint_engine.core.exceptions import MissingDependencyError

    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 256)

    fingerprinter = Fingerprinter(FingerprintConfig())
    image_handler = next(h for h in fingerprinter.handlers if h.name == "image")

    def _raise_missing(_path: Path, *, content: bytes | None = None) -> object:
        raise MissingDependencyError(
            "Pillow is required for image fingerprinting",
            package="Pillow",
            extra="image",
        )

    monkeypatch.setattr(image_handler, "load", _raise_missing)

    with caplog.at_level(logging.WARNING, logger="fingerprint_engine"):
        with pytest.raises(MissingDependencyError):
            fingerprinter.fingerprint_file(path)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("Pillow" in record.getMessage() for record in warnings), [
        record.getMessage() for record in warnings
    ]


def test_batch_skip_emits_warning_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A skipped path in fingerprint_many must emit a WARNING naming the path and
    # error type, in addition to the existing RuntimeWarning.
    good = tmp_path / "a.txt"
    good.write_text("alpha alpha alpha\n" * 40, encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"

    fingerprinter = Fingerprinter(_TEXT_CONFIG)

    with caplog.at_level(logging.WARNING, logger="fingerprint_engine"):
        with pytest.warns(RuntimeWarning):
            fingerprinter.fingerprint_many([good, missing])

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any(
        "does-not-exist.txt" in record.getMessage() and "FileNotFoundError" in record.getMessage()
        for record in warnings
    ), [record.getMessage() for record in warnings]


def test_search_emits_timing_debug_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A search must emit a DEBUG record on the index logger reporting the query
    # hash count, candidate/result counts, and elapsed time.
    path = tmp_path / "sample.py"
    path.write_text("def add(a, b):\n    return a + b\n" * 20, encoding="utf-8")

    fingerprinter = Fingerprinter(_TEXT_CONFIG)
    index = InMemoryHashIndex()
    fingerprint = fingerprinter.fingerprint_file(path)
    index.add(fingerprint)

    with caplog.at_level(logging.DEBUG, logger="fingerprint_engine"):
        index.search(fingerprint, top_k=5)

    debug_records = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG and record.name == "fingerprint_engine.core.index"
    ]
    assert any(
        record.getMessage().startswith("search:") and "results in" in record.getMessage()
        for record in debug_records
    ), [record.getMessage() for record in debug_records]


def test_no_log_output_leaks_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # With no handler configured by the embedding app, the package's NullHandler
    # must absorb records so nothing leaks to stderr during normal use (and no
    # "No handlers could be found" warning is printed).
    path = tmp_path / "sample.py"
    path.write_text("def add(a, b):\n    return a + b\n" * 20, encoding="utf-8")

    # Detach pytest's logging capture so only the package's own handling applies;
    # propagation is on by default, so any leak would reach the root and stderr.
    logging.getLogger("fingerprint_engine").propagate = True

    fingerprinter = Fingerprinter(_TEXT_CONFIG)
    index = InMemoryHashIndex()
    fingerprint = fingerprinter.fingerprint_file(path)
    index.add(fingerprint)
    index.search(fingerprint, top_k=5)

    captured = capsys.readouterr()
    assert "No handlers could be found" not in captured.err
    assert "search:" not in captured.err
