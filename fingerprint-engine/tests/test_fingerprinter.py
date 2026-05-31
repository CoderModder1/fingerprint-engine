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


def _process_pool_available() -> bool:
    """Whether this sandbox can actually spawn a ProcessPoolExecutor worker.

    Some CI/sandbox environments forbid fork/spawn (no /dev/shm, seccomp, etc.).
    The process-mode tests gate on this so they exercise the real parallel path
    where possible and skip with a clear reason where it is impossible -- never
    silently passing.
    """

    from concurrent.futures import ProcessPoolExecutor

    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            return pool.submit(abs, -1).result(timeout=30) == 1
    except Exception:  # noqa: BLE001 - any spawn failure means "unavailable"
        return False


_PROCESS_POOL = _process_pool_available()
_requires_process_pool = pytest.mark.skipif(
    not _PROCESS_POOL,
    reason="ProcessPoolExecutor cannot spawn workers in this environment",
)


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

    def _raise_missing(_path: Path, *, content: bytes | None = None) -> object:
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

    def _raise_decode(_path: Path, *, content: bytes | None = None) -> object:
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

    def _raise_decode(_self: object, _path: Path, *, content: bytes | None = None) -> object:
        raise ValueError("cannot decode")

    for handler in fingerprinter.handlers:
        monkeypatch.setattr(handler, "load", _raise_decode.__get__(handler))

    with pytest.raises(NoHandlerError, match="no handler could fingerprint"):
        fingerprinter.fingerprint_file(path)


def _mixed_batch_config() -> FingerprintConfig:
    # A small explicit window so the batch fingerprints quickly while still
    # exercising the text and binary handlers (a mixed batch).
    return FingerprintConfig(
        window_size=32,
        hop_size=8,
        peak_threshold=0.25,
        peak_percentile=70.0,
        max_delta_t=16,
    )


def _write_mixed_batch(tmp_path: Path) -> list[Path]:
    # Text + binary inputs so both handler routes run through the worker.
    text_a = tmp_path / "alpha.txt"
    text_a.write_text("alpha alpha alpha\n" * 60, encoding="utf-8")
    text_b = tmp_path / "beta.py"
    text_b.write_text(
        "".join(f"def f_{i}(x):\n    return x * {i} + {i * 3}\n\n" for i in range(40)),
        encoding="utf-8",
    )
    payload = tmp_path / "payload.unknown"
    payload.write_bytes(bytes(range(256)) * 8)
    return [text_a, text_b, payload]


@_requires_process_pool
def test_process_mode_matches_thread_mode_hashes_and_order(tmp_path: Path) -> None:
    # The process pool is a pure relocation of CPU-bound work: process mode must
    # return fingerprints with IDENTICAL hashes (and landmarks, ids, handler) in
    # IDENTICAL input order to thread mode, so swapping the parallelism strategy
    # never changes what gets indexed.
    paths = _write_mixed_batch(tmp_path)
    fingerprinter = Fingerprinter(_mixed_batch_config())

    thread_fps = fingerprinter.fingerprint_many(paths, executor="thread")
    process_fps = fingerprinter.fingerprint_many(paths, executor="process")

    # Same order (by name) and same number of results.
    assert [Path(fp.path).name for fp in thread_fps] == [p.name for p in paths]
    assert [Path(fp.path).name for fp in process_fps] == [p.name for p in paths]

    # Byte-identical fingerprints, field by field, in lockstep order.
    assert len(thread_fps) == len(process_fps) == len(paths)
    for thread_fp, process_fp in zip(thread_fps, process_fps, strict=True):
        assert thread_fp.handler == process_fp.handler
        assert thread_fp.file_id == process_fp.file_id
        assert thread_fp.content_sha256 == process_fp.content_sha256
        assert thread_fp.hash_tuples() == process_fp.hash_tuples()
        assert thread_fp.landmarks == process_fp.landmarks
        assert thread_fp.hash_count > 0

    # Mixed batch really exercised more than one handler.
    assert {fp.handler for fp in process_fps} == {"text", "binary"}


@_requires_process_pool
def test_process_mode_default_max_workers_matches_thread_mode(tmp_path: Path) -> None:
    # max_workers=None (the default) must work in process mode too and still
    # produce identical hashes/order -- the pool sizing is irrelevant to output.
    paths = _write_mixed_batch(tmp_path)
    fingerprinter = Fingerprinter(_mixed_batch_config())

    thread_fps = fingerprinter.fingerprint_many(paths)  # default thread, default workers
    process_fps = fingerprinter.fingerprint_many(paths, executor="process")

    assert [Path(fp.path).name for fp in process_fps] == [p.name for p in paths]
    for thread_fp, process_fp in zip(thread_fps, process_fps, strict=True):
        assert thread_fp.hash_tuples() == process_fp.hash_tuples()


@_requires_process_pool
def test_process_mode_fail_soft_skips_bad_path_and_collects_error(tmp_path: Path) -> None:
    # Fail-soft semantics must be identical in process mode: a missing path
    # between good ones is skipped (not fatal), the good fingerprints come back
    # in input order, and the failure is recorded in the errors collector with
    # its real exception type -- no RuntimeWarning when a collector is supplied.
    good_a = tmp_path / "a.txt"
    good_a.write_text("alpha alpha alpha\n" * 40, encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"
    good_b = tmp_path / "b.txt"
    good_b.write_text("bravo bravo bravo\n" * 40, encoding="utf-8")

    fingerprinter = Fingerprinter(_mixed_batch_config())

    collector: list[tuple[str, Exception]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fingerprints = fingerprinter.fingerprint_many(
            [good_a, missing, good_b], errors=collector, executor="process"
        )

    assert [Path(fp.path).name for fp in fingerprints] == ["a.txt", "b.txt"]
    assert len(collector) == 1
    failed_path, failed_exc = collector[0]
    assert failed_path == str(missing)
    assert isinstance(failed_exc, FileNotFoundError)


@_requires_process_pool
def test_process_mode_fail_soft_warns_without_collector(tmp_path: Path) -> None:
    # Without a collector, process mode still emits the RuntimeWarning naming the
    # skipped path (same fallback path as thread mode).
    good_a = tmp_path / "a.txt"
    good_a.write_text("alpha alpha alpha\n" * 40, encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"
    good_b = tmp_path / "b.txt"
    good_b.write_text("bravo bravo bravo\n" * 40, encoding="utf-8")

    fingerprinter = Fingerprinter(_mixed_batch_config())

    with pytest.warns(RuntimeWarning, match=r"skipping .*does-not-exist\.txt"):
        fingerprints = fingerprinter.fingerprint_many(
            [good_a, missing, good_b], executor="process"
        )

    assert [Path(fp.path).name for fp in fingerprints] == ["a.txt", "b.txt"]


@_requires_process_pool
def test_process_mode_raises_on_first_error_when_skip_errors_false(tmp_path: Path) -> None:
    # skip_errors=False must still propagate the first failure in process mode.
    good = tmp_path / "a.txt"
    good.write_text("alpha alpha alpha\n" * 40, encoding="utf-8")
    missing = tmp_path / "does-not-exist.txt"

    fingerprinter = Fingerprinter(_mixed_batch_config())

    with pytest.raises(FileNotFoundError):
        fingerprinter.fingerprint_many(
            [good, missing, good], skip_errors=False, executor="process"
        )


def test_unknown_executor_rejected(tmp_path: Path) -> None:
    # A typo'd executor name fails loudly rather than silently defaulting.
    path = tmp_path / "a.txt"
    path.write_text("alpha alpha\n" * 40, encoding="utf-8")
    fingerprinter = Fingerprinter(_mixed_batch_config())

    with pytest.raises(ValueError, match="unknown executor"):
        fingerprinter.fingerprint_many([path], executor="parallel")  # type: ignore[arg-type]


def test_fingerprinter_is_picklable() -> None:
    # ProcessPoolExecutor under the 'spawn' start method (macOS default) requires
    # everything sent to workers to be picklable. The worker initializer is handed
    # this instance's (config, handlers_package); both must round-trip, and the
    # reconstructed Fingerprinter must produce the same hashes.
    import pickle

    fingerprinter = Fingerprinter(_mixed_batch_config())
    restored = pickle.loads(pickle.dumps(fingerprinter))

    assert [h.name for h in restored.handlers] == [h.name for h in fingerprinter.handlers]
    assert restored.config == fingerprinter.config
    assert restored.handlers_package == fingerprinter.handlers_package


# ---------------------------------------------------------------------------
# A1: single-read pipeline -- the fingerprinted bytes ARE the identity bytes,
# and decoding from the threaded buffer matches decoding from disk.
# ---------------------------------------------------------------------------


def test_fingerprint_file_threads_read_bytes_into_handler(tmp_path: Path) -> None:
    # The fingerprinter reads the file ONCE and hands those exact bytes to the
    # handler, so content_sha256/file_id describe the SAME bytes that were
    # fingerprinted -- no second disk read, no time-of-check/time-of-use window.
    import hashlib

    fingerprinter = Fingerprinter(FingerprintConfig())
    path = tmp_path / "x.py"
    data = ("print('hi')\n" * 64).encode("utf-8")
    path.write_bytes(data)

    text_handler = next(h for h in fingerprinter.handlers if h.name == "text")
    seen: dict[str, object] = {}
    real_load = text_handler.load

    def spy(p: object, *, content: bytes | None = None) -> object:
        seen["content"] = content
        return real_load(p, content=content)

    text_handler.load = spy  # type: ignore[method-assign]
    result = fingerprinter.fingerprint_file(path)

    assert seen["content"] == data  # the handler received the exact file bytes
    assert result.content_sha256 == hashlib.sha256(data).hexdigest()
    # The stored identity is the digest of EXACTLY the bytes that were fingerprinted.
    assert result.content_sha256 == hashlib.sha256(seen["content"]).hexdigest()


def test_handler_load_disk_and_bytes_paths_agree(tmp_path: Path) -> None:
    # Decoding from the threaded bytes (the production single-read path) must be
    # byte-identical to opening the path (the legacy form), for every routed
    # handler -- this is the invariant the A1 refactor rests on.
    fingerprinter = Fingerprinter(FingerprintConfig())

    cases: list[Path] = []
    text = tmp_path / "a.py"
    text.write_text("def f():\n    return 1\n" * 30, encoding="utf-8")
    cases.append(text)
    blob = tmp_path / "a.bin"
    blob.write_bytes(bytes(range(256)) * 16)
    cases.append(blob)
    try:  # image path only when Pillow is installed
        import numpy as np
        from PIL import Image

        img = tmp_path / "a.png"
        rng = np.random.default_rng(7)
        Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8), "RGB").save(img)
        cases.append(img)
    except ImportError:
        pass

    for path in cases:
        content = path.read_bytes()
        candidates = fingerprinter._rank_handlers(path, content[:8192])
        _score, handler = candidates[0]
        pipeline = fingerprinter._handler_pipelines.get(handler.name, fingerprinter.pipeline)
        _l1, from_disk = handler.extract_peaks(handler.to_signal(handler.load(path)), pipeline)
        _l2, from_bytes = handler.extract_peaks(
            handler.to_signal(handler.load(path, content=content)), pipeline
        )
        assert [h.hash_code for h in from_disk] == [h.hash_code for h in from_bytes], path.name


def test_low_max_window_bank_size_does_not_crash_construction() -> None:
    # Regression (v2 review): the per-handler default audio bank (4 windows) must
    # not be gated by the caller's max_window_bank_size -- that cap governs
    # CALLER-supplied global banks. A caller lowering it below 4 previously
    # crashed Fingerprinter() construction for ALL file types.
    for n in (1, 2, 3):
        fingerprinter = Fingerprinter(FingerprintConfig(max_window_bank_size=n))  # must not raise
        audio_pipeline = fingerprinter._handler_pipelines.get("audio")
        assert audio_pipeline is not None
        assert audio_pipeline.config.window_bank == (512, 1024, 2048, 4096)
