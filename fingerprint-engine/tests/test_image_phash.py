"""Tests for the OPT-IN 2D DCT perceptual-hash image mode.

``FingerprintConfig.image_mode`` defaults to ``"raster"`` -- the byte-identical
1D-signal constellation path the image handler has always produced. ``"phash"``
switches to a self-contained DCT perceptual-hash match path that is far more
resize/crop robust. These tests pin three things:

* the default raster path is BYTE-IDENTICAL (same hashes, same metadata strategy)
  -- the opt-in flag must not perturb existing image fingerprints;
* the phash path self-matches and is well-formed (one position-tagged sub-code
  per band, no FFT landmarks);
* phash is measurably MORE robust than raster on at least the resize case
  (the headline claim), exercised through the deterministic accuracy harness.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fingerprinter import Fingerprinter  # noqa: E402 - after sys.path bootstrap
from fingerprint_engine.core.models import FingerprintConfig  # noqa: E402 - after sys.path bootstrap
from fingerprint_engine.handlers.image_handler import (  # noqa: E402 - after sys.path bootstrap
    _PHASH_BANDS,
    _PHASH_BITS,
    compute_phash_bits,
    phash_band_codes,
)

pytest.importorskip("PIL")


def _make_image(rng: np.random.Generator, width: int = 180, height: int = 140):  # noqa: ANN202
    """A reproducible, structured RGB image (gradient + per-file noise/blocks)."""

    from PIL import Image

    yy, xx = np.mgrid[0:height, 0:width]
    base = 128 + 70 * np.sin(2 * np.pi * xx / width) + 50 * np.cos(2 * np.pi * yy / height)
    blocks = (((xx // 17) % 2) ^ ((yy // 13) % 2)) * 40
    noise = rng.normal(0, 6, (height, width))
    arr = np.clip(base + blocks + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")


# ---------------------------------------------------------------------------
# Config flag: validation + default
# ---------------------------------------------------------------------------


def test_image_mode_defaults_to_raster() -> None:
    assert FingerprintConfig().image_mode == "raster"


def test_image_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="image_mode"):
        FingerprintConfig(image_mode="dct").validate()  # type: ignore[arg-type]


def test_configured_image_mode_reaches_discovered_handler() -> None:
    fingerprinter = Fingerprinter(FingerprintConfig(image_mode="phash"))
    image = next(h for h in fingerprinter.handlers if h.name == "image")
    assert image.image_mode == "phash"  # type: ignore[attr-defined]


def test_default_handler_is_raster() -> None:
    fingerprinter = Fingerprinter()
    image = next(h for h in fingerprinter.handlers if h.name == "image")
    assert image.image_mode == "raster"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Default-preserving: raster fingerprints are byte-identical to a fresh default
# ---------------------------------------------------------------------------


def test_default_raster_fingerprint_is_byte_identical(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    path = tmp_path / "pic.png"
    _make_image(rng).save(path)

    # Two independent default fingerprinters must agree exactly, and an
    # explicit image_mode="raster" must equal the implicit default.
    base = Fingerprinter().fingerprint_file(path)
    again = Fingerprinter().fingerprint_file(path)
    explicit = Fingerprinter(FingerprintConfig(image_mode="raster")).fingerprint_file(path)

    assert base.hash_tuples() == again.hash_tuples() == explicit.hash_tuples()
    assert [lm.to_dict() for lm in base.landmarks] == [lm.to_dict() for lm in explicit.landmarks]
    # The raster signal strategy label is unchanged (additive image_mode aside).
    assert base.metadata["signal_strategy"] == "canonical_256x256_grayscale_intensity"
    assert base.metadata["image_mode"] == "raster"
    # Raster keeps the rich multi-thousand-hash constellation.
    assert base.hash_count > 100


# ---------------------------------------------------------------------------
# phash path: well-formed and self-matching
# ---------------------------------------------------------------------------


def test_phash_emits_one_code_per_band(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    path = tmp_path / "pic.png"
    _make_image(rng).save(path)

    fingerprint = Fingerprinter(FingerprintConfig(image_mode="phash")).fingerprint_file(path)
    assert fingerprint.hash_count == _PHASH_BANDS
    # phash is a global descriptor: no spectral landmarks are reported.
    assert fingerprint.landmark_count == 0
    assert fingerprint.metadata["signal_strategy"] == "canonical_256x256_dct_phash"
    assert fingerprint.metadata["image_mode"] == "phash"
    # Bands are tagged by position (freq1 == band index, ascending after sort).
    assert sorted(h.freq1 for h in fingerprint.hashes) == list(range(_PHASH_BANDS))


def test_phash_self_match_is_perfect(tmp_path: Path) -> None:
    from fingerprint_engine.core.index import InMemoryHashIndex

    rng = np.random.default_rng(11)
    fingerprinter = Fingerprinter(FingerprintConfig(image_mode="phash"))
    paths = []
    for i in range(8):
        path = tmp_path / f"img{i}.png"
        _make_image(rng).save(path)
        paths.append(path)
    fingerprints = [fingerprinter.fingerprint_file(p) for p in paths]
    index = InMemoryHashIndex()
    index.add_many(fingerprints)
    for fingerprint in fingerprints:
        results = index.search(fingerprint, top_k=1)
        assert results and results[0].file_id == fingerprint.file_id
        assert results[0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pHash unit properties: deterministic, Hamming-tolerant band collisions
# ---------------------------------------------------------------------------


def test_compute_phash_is_deterministic_and_sized() -> None:
    rng = np.random.default_rng(5)
    grid = rng.integers(0, 256, size=(256, 256)).astype(np.float64)
    bits_a = compute_phash_bits(grid)
    bits_b = compute_phash_bits(grid.copy())
    assert bits_a.shape == (_PHASH_BITS,)
    assert bits_a.dtype == bool
    assert np.array_equal(bits_a, bits_b)


def test_band_codes_tolerate_a_single_bit_flip() -> None:
    rng = np.random.default_rng(13)
    grid = rng.integers(0, 256, size=(256, 256)).astype(np.float64)
    bits = compute_phash_bits(grid)
    codes = phash_band_codes(bits)

    flipped = bits.copy()
    flipped[5] = not flipped[5]  # one bit -> at most one band changes
    flipped_codes = phash_band_codes(flipped)

    shared = sum(1 for a, b in zip(codes, flipped_codes, strict=True) if a == b)
    # A single flipped bit invalidates exactly the one band containing it, so
    # every other band still collides -- the property the Hamming match relies on.
    assert shared >= _PHASH_BANDS - 1


def test_band_codes_are_position_tagged() -> None:
    # Two different bands holding the same bit value must NOT produce the same
    # code (otherwise bands would alias and the offset histogram would mis-vote).
    bits = np.zeros(_PHASH_BITS, dtype=bool)
    bits[0:4] = True  # band 0 set
    codes_a = phash_band_codes(bits)
    bits2 = np.zeros(_PHASH_BITS, dtype=bool)
    bits2[4:8] = True  # band 1 set to the same nibble value
    codes_b = phash_band_codes(bits2)
    assert codes_a[0] != codes_b[1]


# ---------------------------------------------------------------------------
# Robustness: phash beats raster on the resize case (the headline claim)
# ---------------------------------------------------------------------------


def _resize_recall_and_conf(fingerprinter: Fingerprinter, tmp_path: Path) -> tuple[float, float]:
    """Index a small corpus, then measure downscale-resize recall@1 + mean conf."""

    from fingerprint_engine.core.index import InMemoryHashIndex

    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    images = [_make_image(rng) for _ in range(8)]
    paths = []
    for i, image in enumerate(images):
        path = tmp_path / f"c{i}.png"
        image.save(path)
        paths.append(path)
    fingerprints = [fingerprinter.fingerprint_file(p) for p in paths]
    index = InMemoryHashIndex()
    index.add_many(fingerprints)

    hits = 0
    confidences: list[float] = []
    scratch = tmp_path / "_resized.png"
    for i, target in enumerate(fingerprints):
        buffer = io.BytesIO()
        images[i].resize((96, 72)).save(buffer, format="PNG")
        scratch.write_bytes(buffer.getvalue())
        results = index.search(fingerprinter.fingerprint_file(scratch), top_k=1)
        if results and results[0].file_id == target.file_id:
            hits += 1
            confidences.append(results[0].confidence)
    recall = hits / len(fingerprints)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return recall, mean_conf


def test_phash_improves_resize_over_raster(tmp_path: Path) -> None:
    raster_recall, raster_conf = _resize_recall_and_conf(
        Fingerprinter(), tmp_path / "raster"
    )
    phash_recall, phash_conf = _resize_recall_and_conf(
        Fingerprinter(FingerprintConfig(image_mode="phash")), tmp_path / "phash"
    )

    # Both find every resized parent; phash does so with strictly higher
    # confidence (the DCT low-frequency code survives downscaling far better
    # than the row-major raster signal).
    assert raster_recall == 1.0
    assert phash_recall == 1.0
    assert phash_conf > raster_conf
