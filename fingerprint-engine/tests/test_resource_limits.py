"""Resource-limit / untrusted-input hardening for the fingerprinter.

Covers ``FingerprintConfig.max_file_size_bytes`` and the ``FileTooLargeError``
guard: an oversized input must be rejected *before* it is read into memory,
must be skipped (not fatal) by the fail-soft batch path, and ``validate()`` must
reject negative knobs.
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import FileTooLargeError, FingerprintError
from fingerprint_engine.core.fingerprinter import Fingerprinter as _Fingerprinter
from fingerprint_engine.core.models import FingerprintConfig

# Snapshot the engine modules as imported HERE (at collection time, before any
# test runs). ``test_packaging`` evicts every ``fingerprint_engine.*`` entry
# from ``sys.modules``; re-instating this exact set before each test keeps the
# classes our top-level imports reference live and self-consistent.
_ENGINE_MODULES = {
    name: mod for name, mod in sys.modules.items() if name.startswith("fingerprint_engine")
}

# A small window so a tiny text file still yields searchable hashes.
_SMALL_WINDOW = FingerprintConfig(
    window_size=32,
    hop_size=8,
    peak_threshold=0.25,
    peak_percentile=70.0,
    max_peaks_per_frame=5,
    constellation_fanout=4,
    max_delta_t=16,
)


@pytest.fixture(autouse=True)
def _restore_engine_modules() -> Iterator[None]:
    """Pin the ``fingerprint_engine.*`` modules to the ones imported here.

    ``test_packaging`` deletes every ``fingerprint_engine.*`` entry from
    ``sys.modules`` (without restoring them) to assert lazy-import hygiene. If
    that test ran first, our top-level imports (``_Fingerprinter``,
    ``FileTooLargeError``, ...) would be left pointing at evicted modules, and
    constructing a ``Fingerprinter`` would re-import the handlers against a
    *fresh* ``FileHandler`` class -- so ``issubclass`` fails ("no file handlers
    discovered") and ``FileTooLargeError`` identities diverge. Re-instating the
    exact modules this file imported keeps one self-consistent class identity,
    making these tests independent of collection/run order.
    """

    sys.modules.update(_ENGINE_MODULES)
    yield


@pytest.fixture
def fingerprinter_cls() -> type[_Fingerprinter]:
    return _Fingerprinter


def _write_text(path: Path) -> Path:
    path.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "for i in range(20):\n"
        "    print(add(i, i * 2))\n",
        encoding="utf-8",
    )
    return path


def test_file_over_limit_raises_file_too_large(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    path = _write_text(tmp_path / "sample.py")
    size = path.stat().st_size
    assert size > 10  # the source above is comfortably larger than the limit

    config = dataclasses.replace(_SMALL_WINDOW, max_file_size_bytes=10)
    fingerprinter = fingerprinter_cls(config)

    with pytest.raises(FileTooLargeError) as excinfo:
        fingerprinter.fingerprint_file(path)

    exc = excinfo.value
    assert isinstance(exc, FingerprintError)
    assert exc.limit == 10
    assert exc.size == size
    assert "exceeds max_file_size_bytes" in str(exc)


def test_oversized_file_is_rejected_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fingerprinter_cls: type[_Fingerprinter],
) -> None:
    # The size check must short-circuit BEFORE the whole file is read into
    # memory. If Path.read_bytes is ever reached for an oversized file the OOM
    # guard is defeated, so make it explode and assert it is never called.
    path = _write_text(tmp_path / "sample.py")

    def _boom(self: Path) -> bytes:  # pragma: no cover - must not run
        raise AssertionError("read_bytes() was called before the size check")

    monkeypatch.setattr(Path, "read_bytes", _boom)

    config = dataclasses.replace(_SMALL_WINDOW, max_file_size_bytes=10)
    fingerprinter = fingerprinter_cls(config)

    with pytest.raises(FileTooLargeError):
        fingerprinter.fingerprint_file(path)


def test_file_under_limit_is_fingerprinted(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    path = _write_text(tmp_path / "sample.py")
    size = path.stat().st_size

    config = dataclasses.replace(_SMALL_WINDOW, max_file_size_bytes=size + 1)
    fingerprinter = fingerprinter_cls(config)

    fingerprint = fingerprinter.fingerprint_file(path)
    assert fingerprint.handler == "text"
    assert fingerprint.size_bytes == size
    assert fingerprint.hash_count > 0


def test_zero_limit_means_unlimited(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    path = _write_text(tmp_path / "sample.py")

    config = dataclasses.replace(_SMALL_WINDOW, max_file_size_bytes=0)
    fingerprinter = fingerprinter_cls(config)

    # A 0 limit must not stat-reject; any normal file fingerprints fine.
    fingerprint = fingerprinter.fingerprint_file(path)
    assert fingerprint.hash_count > 0


def test_batch_skips_oversized_file_and_keeps_good_ones(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    small = _write_text(tmp_path / "small.py")
    big = tmp_path / "big.py"
    # 'big' is the same hashable source plus padding, so it still fingerprints
    # on its own merits -- it is rejected purely for exceeding the size limit.
    big.write_text(small.read_text(encoding="utf-8") + "# pad\n" * 200, encoding="utf-8")

    # Limit sits between the two files: 'big' is rejected, 'small' goes through.
    limit = small.stat().st_size + 1
    assert big.stat().st_size > limit

    config = dataclasses.replace(_SMALL_WINDOW, max_file_size_bytes=limit)
    fingerprinter = fingerprinter_cls(config)

    errors: list[tuple[str, Exception]] = []
    results = fingerprinter.fingerprint_many([big, small], errors=errors)

    # The oversized file is skipped (fail-soft), not fatal; the good one survives.
    assert [fp.path for fp in results] == [str(small.resolve())]
    assert len(errors) == 1
    skipped_path, skipped_exc = errors[0]
    assert skipped_path == str(big)
    assert isinstance(skipped_exc, FileTooLargeError)


def test_batch_emits_warning_when_no_error_collector(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    big = _write_text(tmp_path / "big.py")
    config = dataclasses.replace(_SMALL_WINDOW, max_file_size_bytes=10)
    fingerprinter = fingerprinter_cls(config)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = fingerprinter.fingerprint_many([big])

    assert results == []
    assert any(
        issubclass(w.category, RuntimeWarning) and "FileTooLargeError" in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"max_file_size_bytes": -1}, "max_file_size_bytes must be non-negative"),
        ({"max_pdf_pages": -1}, "max_pdf_pages must be non-negative"),
        ({"max_image_pixels": -1}, "max_image_pixels must be non-negative"),
    ],
)
def test_validate_rejects_negative_limits(
    overrides: dict[str, object], match: str
) -> None:
    config = dataclasses.replace(FingerprintConfig(), **overrides)
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_default_limits_are_finite_and_sane() -> None:
    config = FingerprintConfig()
    # A finite default bounds the OOM vector...
    assert config.max_file_size_bytes == 256 * 1024 * 1024
    assert config.max_file_size_bytes > 0
    # ...while max_pdf_pages defaults to unlimited (handler enforces it).
    assert config.max_pdf_pages == 0
    # ...and a finite default pixel cap bounds the image decode-bomb vector.
    assert config.max_image_pixels == 89_478_485
    assert config.max_image_pixels > 0
    config.validate()


def _write_noise_png(path: Path, size: tuple[int, int]) -> Path:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return path


def test_image_over_pixel_cap_raises_and_batch_skips_it(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    # B1: a compressed image that decodes past max_image_pixels must be rejected
    # BEFORE decode (the on-disk size cap cannot see decoded resolution), and the
    # rejection must FAIL LOUD -- not silently demote to a binary fingerprint.
    pytest.importorskip("PIL", exc_type=ImportError)
    path = _write_noise_png(tmp_path / "img.png", (64, 64))  # 4096 px

    config = dataclasses.replace(FingerprintConfig(), max_image_pixels=100)
    fingerprinter = fingerprinter_cls(config)
    with pytest.raises(FileTooLargeError) as excinfo:
        fingerprinter.fingerprint_file(path)
    assert excinfo.value.size == 4096
    assert excinfo.value.limit == 100

    # Fail-soft batch SKIPS it (records the error) instead of producing a binary
    # fingerprint of the same bytes.
    errors: list[tuple[str, Exception]] = []
    results = fingerprinter.fingerprint_many([path], errors=errors)
    assert results == []
    assert errors and isinstance(errors[0][1], FileTooLargeError)


def test_image_under_pixel_cap_fingerprints_as_image(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    pytest.importorskip("PIL", exc_type=ImportError)
    path = _write_noise_png(tmp_path / "img.png", (64, 64))

    # The default cap (~89.5 Mpx) and unlimited (0) both fingerprint a normal image.
    for cap in (FingerprintConfig().max_image_pixels, 0):
        fingerprinter = fingerprinter_cls(
            dataclasses.replace(FingerprintConfig(), max_image_pixels=cap)
        )
        assert fingerprinter.fingerprint_file(path).handler == "image"


def test_file_content_sha256_caps_and_rejects_non_regular(tmp_path: Path) -> None:
    # B2: the cheap-sha helper used by incremental ingest must enforce the cap on
    # the bytes (not just trust stat) and reject non-regular inputs (a FIFO/dir
    # reports st_size==0 and an unbounded read would block forever).
    import hashlib

    from fingerprint_engine.core.fingerprinter import file_content_sha256

    path = tmp_path / "f.bin"
    data = b"abc" * 100
    path.write_bytes(data)

    # Correct digest with no cap and with a cap exactly at the size.
    assert file_content_sha256(path) == hashlib.sha256(data).hexdigest()
    assert (
        file_content_sha256(path, max_file_size_bytes=len(data))
        == hashlib.sha256(data).hexdigest()
    )
    # A file over the cap raises (the stat fast-path).
    with pytest.raises(FileTooLargeError):
        file_content_sha256(path, max_file_size_bytes=len(data) - 1)
    # A non-regular input (a directory) is refused outright.
    with pytest.raises(OSError):
        file_content_sha256(tmp_path)


def test_image_pixel_cap_reaches_phash_handler(
    tmp_path: Path, fingerprinter_cls: type[_Fingerprinter]
) -> None:
    # configure() wires the cap to the discovered handler in BOTH image modes,
    # so the opt-in phash handler (which inherits load()) is guarded too.
    pytest.importorskip("PIL", exc_type=ImportError)
    path = _write_noise_png(tmp_path / "img.png", (64, 64))

    config = dataclasses.replace(
        FingerprintConfig(), image_mode="phash", max_image_pixels=100
    )
    fingerprinter = fingerprinter_cls(config)
    with pytest.raises(FileTooLargeError):
        fingerprinter.fingerprint_file(path)
