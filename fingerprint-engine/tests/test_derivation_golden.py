"""Derivation guard: operational/limit config fields must NOT alter the hashes.

Determinism (same bytes -> same hashes) is covered in ``test_fingerprinter.py``,
and refactor byte-identity by the manual ``benchmarks/hash_capture.py`` gate.
This pins the complementary freeze invariant: the resource-limit / operational
``FingerprintConfig`` fields (size caps, etc.) are NOT part of the hash
derivation, so changing them leaves the default hashes byte-identical for inputs
under the caps. If a future "operational" field silently leaks into the
derivation, this fails -- which, once 1.0 freezes the default derivation, would
otherwise be an undetected MAJOR break (every shipped index would be wrong).

It deliberately varies ONLY operational fields, never a tuning/derivation knob
(``window_size``, ``hop_size``, ``peak_*``, ``constellation_fanout``,
``hash_bits``, ``freq_quantization``, ``window_bank``, ``image_mode``, ...),
which are expected to change hashes by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.models import FingerprintConfig

# Changes ONLY operational/limit fields to non-default values that are inert for
# the text/binary fixtures below: the size caps sit far above these tiny inputs,
# max_window_bank_size is inert while window_bank stays the default None, and
# max_pdf_pages applies only to the PDF handler.
_OPERATIONAL_ONLY = FingerprintConfig(
    max_file_size_bytes=512 * 1024 * 1024,
    max_signal_samples=4_000_000,
    max_pdf_pages=10,
    max_image_pixels=200_000_000,
    max_window_bank_size=8,
)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("note.txt", b"The quick brown fox jumps over the lazy dog 0123456789.\n" * 60),
        ("blob.bin", bytes((i * 37 + 11) % 256 for i in range(8192))),
    ],
)
def test_operational_fields_do_not_change_derivation(
    tmp_path: Path, name: str, content: bytes
) -> None:
    path = tmp_path / name
    path.write_bytes(content)

    default_fp = Fingerprinter().fingerprint_file(path)
    operational_fp = Fingerprinter(_OPERATIONAL_ONLY).fingerprint_file(path)

    # Same routed handler and byte-identical (hash_code, time_offset) sequence.
    assert default_fp.handler == operational_fp.handler
    assert default_fp.hash_tuples() == operational_fp.hash_tuples()
    # And the fixture is non-trivial, so the equality above is meaningful.
    assert default_fp.hash_count > 0
