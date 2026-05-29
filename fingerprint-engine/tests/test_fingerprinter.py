from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import FingerprintConfig


def test_text_fingerprint_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "for i in range(20):\n"
        "    print(add(i, i * 2))\n",
        encoding="utf-8",
    )
    fingerprinter = Fingerprinter(
        FingerprintConfig(
            window_size=32,
            hop_size=8,
            peak_threshold=0.25,
            peak_percentile=70.0,
            max_peaks_per_frame=5,
            constellation_fanout=4,
            max_delta_t=16,
        )
    )

    first = fingerprinter.fingerprint_file(path)
    second = fingerprinter.fingerprint_file(path)

    assert first.handler == "text"
    assert first.hash_tuples() == second.hash_tuples()
    assert first.landmarks == second.landmarks
    assert first.hash_count > 0


def test_binary_fallback_handles_unknown_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.unknown"
    path.write_bytes(bytes(range(256)) * 8)
    fingerprinter = Fingerprinter(
        FingerprintConfig(
            window_size=64,
            hop_size=16,
            peak_threshold=0.4,
            peak_percentile=75.0,
            max_delta_t=12,
        )
    )

    fingerprint = fingerprinter.fingerprint_file(path)

    assert fingerprint.handler == "binary"
    assert fingerprint.hash_count > 0


def test_batch_processing_preserves_order(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"file-{index}.txt"
        path.write_text(f"hello {index}\n" * 40, encoding="utf-8")
        paths.append(path)
    fingerprinter = Fingerprinter(
        FingerprintConfig(
            window_size=32,
            hop_size=8,
            peak_threshold=0.25,
            peak_percentile=70.0,
            max_delta_t=16,
        )
    )

    fingerprints = fingerprinter.fingerprint_many(paths, max_workers=2)

    assert [Path(item.path).name for item in fingerprints] == [path.name for path in paths]
    assert all(item.handler == "text" for item in fingerprints)


def test_text_prefix_matches_full_file(tmp_path: Path) -> None:
    # Scale-invariance regression: a truncated copy must still match its parent.
    # Sequence handlers use a fixed window, so the prefix's hashes are a subset
    # of the parent's instead of landing on a different length-adaptive window.
    lines = [f"def function_{i}(value):\n    return value * {i} + {i * 7}\n\n" for i in range(80)]
    full = tmp_path / "full.py"
    full.write_text("".join(lines), encoding="utf-8")
    prefix = tmp_path / "prefix.py"
    prefix.write_text("".join(lines[:48]), encoding="utf-8")  # first 60%

    fp = Fingerprinter(FingerprintConfig())
    index = InMemoryHashIndex()
    index.add(fp.fingerprint_file(full))

    results = index.search(fp.fingerprint_file(prefix), top_k=1)

    assert results, "truncated file should still match its parent"
    assert Path(results[0].metadata["path"]).name == "full.py"
    assert results[0].aligned_votes > 50  # a coherent match, not a single stray vote


def test_resized_image_matches_original(tmp_path: Path) -> None:
    # Scale-invariance regression: the same picture at a different resolution
    # must still match, thanks to canonical-size normalisation in the handler.
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:256, 0:256]
    base = ((xx + yy) / 512.0 * 200 + rng.integers(0, 40, size=(256, 256))).astype(np.uint8)
    original = tmp_path / "image.png"
    Image.fromarray(base, mode="L").save(original)
    smaller = tmp_path / "image_small.png"
    Image.fromarray(base, mode="L").resize((180, 180)).save(smaller)

    fp = Fingerprinter(FingerprintConfig())
    index = InMemoryHashIndex()
    index.add(fp.fingerprint_file(original))

    results = index.search(fp.fingerprint_file(smaller), top_k=1)

    assert results, "resized image should still match the original"
    assert Path(results[0].metadata["path"]).name == "image.png"
    assert results[0].aligned_votes > 50


def test_fingerprint_records_effective_window(tmp_path: Path) -> None:
    # The window actually used (possibly adapted for short input) must be
    # recorded so adaptive behaviour is transparent and reproducible.
    path = tmp_path / "short.txt"
    path.write_text("The quick brown fox jumps over the lazy dog.\n" * 12, encoding="utf-8")
    fingerprinter = Fingerprinter(FingerprintConfig())  # default window 4096

    fingerprint = fingerprinter.fingerprint_file(path)

    assert "effective_window_size" in fingerprint.metadata
    assert "effective_hop_size" in fingerprint.metadata
    effective_window = fingerprint.metadata["effective_window_size"]
    # Short input -> window adapted below the configured 4096.
    assert effective_window <= fingerprint.config["window_size"]
    assert fingerprint.metadata["effective_hop_size"] >= 1


def test_empty_file_warns_about_unsearchable_fingerprint(tmp_path: Path) -> None:
    # A featureless file cannot yield constellation pairs even with adaptive
    # windowing; the engine must warn rather than fail silently.
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    fingerprinter = Fingerprinter(FingerprintConfig())

    with pytest.warns(RuntimeWarning, match="unsearchable"):
        fingerprint = fingerprinter.fingerprint_file(path)

    assert fingerprint.hash_count == 0
