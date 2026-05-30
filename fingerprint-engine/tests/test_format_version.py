"""Tests for the ``fingerprint_format_version`` mechanism.

The format version stamps the HASH-DERIVATION format onto a fingerprint and into
an index/snapshot (distinct from the snapshot ``schema_version``), WITHOUT
changing any hash code or search ranking. These tests pin three properties:

1. the version is present and stable at the default config (and the default
   fingerprint/search path is byte-identical -- no version-driven behavior
   change, no warning);
2. a cross-format query against an index is detected (RuntimeWarning by default,
   a dedicated exception under ``strict_format=True``);
3. each opt-in HASH-CHANGING flag records a *different* effective version, so an
   index built with one is detectably incompatible with a default index.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import (
    FingerprintError,
    FormatVersionMismatchError,
)
from fingerprint_engine.core.index import (
    SNAPSHOT_FORMAT_VERSION_KEY,
    SNAPSHOT_SCHEMA_VERSION,
    InMemoryHashIndex,
)
from fingerprint_engine.core.models import (
    FINGERPRINT_FORMAT_VERSION,
    FORMAT_VERSION_KEY,
    ConstellationHash,
    Fingerprint,
    FingerprintConfig,
    effective_format_version,
)


def make_fingerprint(
    file_id: str,
    codes: list[int],
    *,
    version: int = FINGERPRINT_FORMAT_VERSION,
) -> Fingerprint:
    """A fingerprint whose recorded format version is ``version``."""

    hashes = [
        ConstellationHash(
            hash_code=code,
            time_offset=index,
            anchor_time=index,
            target_time=index,
            freq1=0,
            freq2=0,
            delta_t=0,
        )
        for index, code in enumerate(codes)
    ]
    return Fingerprint(
        file_id=file_id,
        path=f"/tmp/{file_id}",
        handler="test",
        size_bytes=len(codes),
        content_sha256=file_id,
        config={FORMAT_VERSION_KEY: version},
        hashes=hashes,
    )


# --- 1. present + stable at the default; default path byte-identical ----------


def test_default_config_effective_version_is_baseline() -> None:
    assert effective_format_version(FingerprintConfig()) == FINGERPRINT_FORMAT_VERSION


def test_fingerprinter_stamps_default_format_version(tmp_path: Path) -> None:
    # A real default-config fingerprint carries the baseline version in config,
    # additively (the tuning keys are untouched), and exposes it via the property.
    from fingerprint_engine.core.fingerprinter import Fingerprinter

    sample = tmp_path / "a.txt"
    sample.write_bytes(b"The quick brown fox jumps over the lazy dog. " * 200)
    fingerprint = Fingerprinter().fingerprint_file(sample)

    assert fingerprint.config[FORMAT_VERSION_KEY] == FINGERPRINT_FORMAT_VERSION
    assert fingerprint.format_version == FINGERPRINT_FORMAT_VERSION
    # Additive: the stamp does not displace the tuning parameters.
    assert "window_size" in fingerprint.config
    assert "freq_quantization" in fingerprint.config


def test_fingerprint_without_stamp_defaults_to_baseline() -> None:
    # Legacy / snapshot-rebuilt fingerprints have no stamp -> treated as default.
    legacy = Fingerprint(
        file_id="x",
        path="/x",
        handler="test",
        size_bytes=1,
        content_sha256="x",
        config={},
    )
    assert legacy.format_version == FINGERPRINT_FORMAT_VERSION


def test_default_search_path_unchanged_and_silent() -> None:
    # A same-format query produces a match with NO warning -- the new check is a
    # no-op on the default path (rankings byte-identical to before it existed).
    index = InMemoryHashIndex()
    index.add(make_fingerprint("a", [10, 20, 30]))
    assert index.format_version == FINGERPRINT_FORMAT_VERSION

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        results = index.search(make_fingerprint("q", [10, 20, 30]))
    assert results and results[0].file_id == "a"


# --- 2. mismatch detection: warn by default, raise under strict ---------------


def test_cross_format_query_warns_by_default() -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("a", [10, 20, 30], version=1))

    with pytest.warns(RuntimeWarning, match="format version"):
        index.search(make_fingerprint("q", [10, 20, 30], version=1001))


def test_cross_format_query_raises_under_strict() -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("a", [10, 20, 30], version=1))

    with pytest.raises(FormatVersionMismatchError) as excinfo:
        index.search(
            make_fingerprint("q", [10, 20, 30], version=1001), strict_format=True
        )
    assert excinfo.value.query_version == 1001
    assert excinfo.value.index_version == 1
    # Joins the FingerprintError family for unified handling.
    assert isinstance(excinfo.value, FingerprintError)


def test_cross_format_add_warns_and_first_writer_wins() -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("a", [10], version=1))

    with pytest.warns(RuntimeWarning, match="do not share a code space"):
        index.add(make_fingerprint("b", [20], version=2001))
    # The index keeps its pinned version; a stray add never re-stamps the corpus.
    assert index.format_version == 1


def test_empty_index_search_does_not_warn_on_any_version() -> None:
    # An empty/unpinned index reports the baseline and is compatible with a
    # baseline query (no postings to be incompatible with).
    index = InMemoryHashIndex()
    assert index.format_version == FINGERPRINT_FORMAT_VERSION
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert index.search(make_fingerprint("q", [1, 2, 3])) == []


# --- snapshot round-trip carries the version ---------------------------------


def test_snapshot_round_trips_non_default_format_version(tmp_path: Path) -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("a", [1, 2, 3], version=4001))
    path = tmp_path / "index.json"
    index.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    # Both versions are present and distinct concerns.
    assert raw[SNAPSHOT_FORMAT_VERSION_KEY] == 4001
    assert raw["schema_version"] == SNAPSHOT_SCHEMA_VERSION

    loaded = InMemoryHashIndex.load(path)
    assert loaded.format_version == 4001
    # A default-version query against the loaded 4001 index is now a mismatch.
    with pytest.warns(RuntimeWarning, match="format version"):
        loaded.search(make_fingerprint("q", [1, 2, 3], version=1))


def test_default_snapshot_stamps_baseline_version(tmp_path: Path) -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("a", [1, 2, 3]))
    path = tmp_path / "index.json"
    index.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw[SNAPSHOT_FORMAT_VERSION_KEY] == FINGERPRINT_FORMAT_VERSION


def test_legacy_snapshot_without_field_loads_as_default(tmp_path: Path) -> None:
    # A snapshot written before the field existed has no format-version key; it
    # must still load and be treated as the default version (not rejected).
    payload = {
        "backend": "in_memory",
        "files": {"a": [[1, 0], [2, 0]]},
        "metadata": {"a": {"handler": "test", "hash_count": 2}},
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert SNAPSHOT_FORMAT_VERSION_KEY not in payload

    loaded = InMemoryHashIndex.load(path)
    assert loaded.format_version == FINGERPRINT_FORMAT_VERSION


def test_from_dict_restores_format_version() -> None:
    payload = {
        "backend": "in_memory",
        "files": {"a": [[1, 0]]},
        "metadata": {},
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        SNAPSHOT_FORMAT_VERSION_KEY: 2001,
    }
    index = InMemoryHashIndex.from_dict(payload)
    assert index.format_version == 2001


# --- 3. each opt-in hash-changer records a different effective version --------


@pytest.mark.parametrize(
    "overrides",
    [
        {"freq_quantization": 4},
        {"window_bank": (512, 1024)},
        {"image_mode": "phash"},
    ],
)
def test_each_opt_in_flag_bumps_effective_version(overrides: dict) -> None:
    config = dataclasses.replace(FingerprintConfig(), **overrides)
    assert effective_format_version(config) != FINGERPRINT_FORMAT_VERSION


def test_opt_in_flags_record_distinct_versions() -> None:
    # The default and each single-flag config land on distinct versions, so an
    # index built with one flag is detectable both from a default index and from
    # an index built with a different flag.
    versions = {
        effective_format_version(FingerprintConfig()),
        effective_format_version(
            dataclasses.replace(FingerprintConfig(), freq_quantization=4)
        ),
        effective_format_version(
            dataclasses.replace(FingerprintConfig(), window_bank=(512, 1024))
        ),
        effective_format_version(
            dataclasses.replace(FingerprintConfig(), image_mode="phash")
        ),
    }
    assert len(versions) == 4


def test_combined_flags_distinct_from_any_single_flag() -> None:
    # Enabling several flags lands on a value distinct from any one alone, so a
    # multi-flag index is not mistaken for a single-flag one.
    base = effective_format_version(FingerprintConfig())
    quant = effective_format_version(
        dataclasses.replace(FingerprintConfig(), freq_quantization=4)
    )
    bank = effective_format_version(
        dataclasses.replace(FingerprintConfig(), window_bank=(512, 1024))
    )
    both = effective_format_version(
        dataclasses.replace(
            FingerprintConfig(), freq_quantization=4, window_bank=(512, 1024)
        )
    )
    assert both not in {base, quant, bank}


def test_opt_in_flag_index_detectably_incompatible_with_default() -> None:
    # An index whose postings were derived with an opt-in flag (here simulated by
    # stamping the flag's effective version) flags a default-config query.
    quant_version = effective_format_version(
        dataclasses.replace(FingerprintConfig(), freq_quantization=4)
    )
    index = InMemoryHashIndex()
    index.add(make_fingerprint("a", [10, 20], version=quant_version))
    assert index.format_version == quant_version

    with pytest.raises(FormatVersionMismatchError):
        index.search(
            make_fingerprint("q", [10, 20], version=FINGERPRINT_FORMAT_VERSION),
            strict_format=True,
        )
