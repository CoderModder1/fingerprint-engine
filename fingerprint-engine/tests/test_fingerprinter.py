from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import MissingDependencyError, NoHandlerError
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


def test_batch_skips_bad_paths_by_default_and_warns(tmp_path: Path) -> None:
    # Fail-soft default: a missing path between two good ones is skipped (with a
    # RuntimeWarning naming it) and the good fingerprints come back in input order.
    good_a = tmp_path / "a.txt"
    good_a.write_text("alpha alpha alpha\n" * 40, encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"
    good_b = tmp_path / "b.txt"
    good_b.write_text("bravo bravo bravo\n" * 40, encoding="utf-8")

    fingerprinter = Fingerprinter(
        FingerprintConfig(
            window_size=32,
            hop_size=8,
            peak_threshold=0.25,
            peak_percentile=70.0,
            max_delta_t=16,
        )
    )

    with pytest.warns(RuntimeWarning, match=r"skipping .*does-not-exist\.txt"):
        fingerprints = fingerprinter.fingerprint_many([good_a, missing, good_b])

    assert [Path(item.path).name for item in fingerprints] == ["a.txt", "b.txt"]


def test_batch_collects_structured_errors(tmp_path: Path) -> None:
    # With an `errors` collector and skip_errors=True, each failure appends a
    # (str(path), exc) tuple (in input order) instead of emitting a warning, so
    # callers get the real exception type/message without parsing text.
    good_a = tmp_path / "a.txt"
    good_a.write_text("alpha alpha alpha\n" * 40, encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"
    good_b = tmp_path / "b.txt"
    good_b.write_text("bravo bravo bravo\n" * 40, encoding="utf-8")

    fingerprinter = Fingerprinter(
        FingerprintConfig(
            window_size=32,
            hop_size=8,
            peak_threshold=0.25,
            peak_percentile=70.0,
            max_delta_t=16,
        )
    )

    collector: list[tuple[str, Exception]] = []
    # No RuntimeWarning is emitted when a collector is supplied.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fingerprints = fingerprinter.fingerprint_many(
            [good_a, missing, good_b], errors=collector
        )

    assert [Path(item.path).name for item in fingerprints] == ["a.txt", "b.txt"]
    assert len(collector) == 1
    failed_path, failed_exc = collector[0]
    assert failed_path == str(missing)
    assert isinstance(failed_exc, FileNotFoundError)


def test_batch_raises_on_first_error_when_skip_errors_false(tmp_path: Path) -> None:
    # Legacy behavior is preserved with skip_errors=False: the first failure
    # (here a missing file) propagates instead of being swallowed.
    good = tmp_path / "a.txt"
    good.write_text("alpha alpha alpha\n" * 40, encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"

    fingerprinter = Fingerprinter(
        FingerprintConfig(
            window_size=32,
            hop_size=8,
            peak_threshold=0.25,
            peak_percentile=70.0,
            max_delta_t=16,
        )
    )

    with pytest.raises(FileNotFoundError):
        fingerprinter.fingerprint_many([good, missing, good], skip_errors=False)


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


def test_text_differing_lengths_share_window_and_match(tmp_path: Path) -> None:
    # Defect B behavioral regression. Two copies of the SAME content at DIFFERENT
    # lengths -- both comfortably above the fixed window's tiny floor (>= 2 frames
    # at 512/128, i.e. >= 640 chars) -- must:
    #   (a) use the SAME effective window (the fixed 512 the text handler declares)
    #   (b) so the shorter file's hashes strongly overlap the longer's -> they MATCH.
    #
    # This FAILS on the length-adaptive code: the longer file (~2.3k chars) maps to
    # window 256 and the shorter (~1.1k chars) to window 128, two different time
    # grids whose hashes never align (zero overlap, no match).
    lines = [f"value_{i} = compute({i}, {i * 3}) + offset_{i}\n" for i in range(60)]
    long_path = tmp_path / "long.txt"
    long_path.write_text("".join(lines), encoding="utf-8")  # ~2.3k chars
    short_path = tmp_path / "short.txt"
    short_path.write_text("".join(lines[:30]), encoding="utf-8")  # ~1.1k chars, different length

    fp = Fingerprinter(FingerprintConfig())  # default config -> text handler window 512/hop 128
    assert fp._handler_pipelines["text"].config.window_size == 512

    long_fp = fp.fingerprint_file(long_path)
    short_fp = fp.fingerprint_file(short_path)

    # (a) same effective window despite the different lengths.
    assert long_fp.handler == short_fp.handler == "text"
    assert long_fp.metadata["effective_window_size"] == short_fp.metadata["effective_window_size"]
    assert short_fp.hash_count > 0

    # (b) the shorter file's hash codes are (almost) a subset of the longer's.
    long_codes = {item.hash_code for item in long_fp.hashes}
    short_codes = {item.hash_code for item in short_fp.hashes}
    overlap = long_codes & short_codes
    assert len(overlap) / len(short_codes) > 0.5

    # ... and that overlap is enough for a coherent index match, not a stray vote.
    index = InMemoryHashIndex()
    index.add(long_fp)
    results = index.search(short_fp, top_k=1)
    assert results, "shorter same-content file should match the longer one"
    assert Path(results[0].metadata["path"]).name == "long.txt"
    assert results[0].aligned_votes > 50


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


def test_missing_dependency_does_not_fall_back_to_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A missing optional dependency in the correct handler must fail loud
    # instead of silently demoting to the binary handler, which would produce
    # raw-byte hashes that are incomparable to those made with the dependency
    # installed (silent index corruption).
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 256)

    fingerprinter = Fingerprinter(FingerprintConfig())

    image_handler = next(h for h in fingerprinter.handlers if h.name == "image")

    def _raise_missing(_path: Path) -> object:
        raise MissingDependencyError(
            "Pillow is required for image fingerprinting",
            package="Pillow",
            extra="image",
        )

    monkeypatch.setattr(image_handler, "load", _raise_missing)

    with pytest.raises(MissingDependencyError) as excinfo:
        fingerprinter.fingerprint_file(path)

    assert excinfo.value.package == "Pillow"
    assert excinfo.value.extra == "image"


def test_decode_error_falls_through_to_other_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A genuine "this handler cannot decode this content" error must still fall
    # through to the next candidate (unlike a missing dependency).
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096)

    fingerprinter = Fingerprinter(FingerprintConfig())
    image_handler = next(h for h in fingerprinter.handlers if h.name == "image")

    def _raise_decode(_path: Path) -> object:
        raise ValueError("cannot decode image content")

    monkeypatch.setattr(image_handler, "load", _raise_decode)

    fingerprint = fingerprinter.fingerprint_file(path)

    # Image handler failed to decode -> demoted to the binary fallback.
    assert fingerprint.handler == "binary"


def test_no_handler_error_when_all_candidates_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When every candidate handler raises a non-dependency error, the engine
    # raises NoHandlerError with the aggregated messages.
    path = tmp_path / "payload.unknown"
    path.write_bytes(bytes(range(256)) * 8)

    fingerprinter = Fingerprinter(FingerprintConfig())

    def _raise_decode(_self: object, _path: Path) -> object:
        raise ValueError("cannot decode")

    for handler in fingerprinter.handlers:
        monkeypatch.setattr(handler, "load", _raise_decode.__get__(handler))

    with pytest.raises(NoHandlerError, match="no handler could fingerprint"):
        fingerprinter.fingerprint_file(path)
