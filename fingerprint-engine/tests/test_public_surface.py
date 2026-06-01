"""Freeze guard: pin the public surface so any drift trips CI.

VERSIONING.md declares the stable public interface to be `fingerprint_engine`'s
`__all__` plus the layout of the public model dataclasses. These tests pin both
so that an accidental addition/removal/reorder -- which would be a silent MAJOR
break once 1.0 freezes the surface -- fails loudly instead. Update the expected
sets here *deliberately* (and bump the version per the SemVer policy) when the
surface is intended to change.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fingerprint_engine as fe
from fingerprint_engine.core.models import (
    Calibration,
    IndexPosting,
    SearchResult,
)

# The exact set of names re-exported from the package. This IS the public API
# per VERSIONING.md §"Stable public interfaces".
_EXPECTED_ALL = {
    "FINGERPRINT_FORMAT_VERSION",
    "Calibration",
    "ConstellationHash",
    "DedupReport",
    "ExactDuplicateCluster",
    "Fingerprint",
    "FingerprintConfig",
    "FingerprintError",
    "Fingerprinter",
    "FormatVersionMismatchError",
    "HashIndex",
    "InMemoryHashIndex",
    "IndexPosting",
    "InvalidSnapshotError",
    "LandmarkPoint",
    "MissingDependencyError",
    "NearDuplicateCluster",
    "NoHandlerError",
    "PostgresHashIndex",
    "RedisHashIndex",
    "SQLiteHashIndex",
    "SearchResult",
    "SnapshotWriteRefused",
    "__version__",
    "effective_format_version",
    "find_duplicates",
}


def test_all_membership_is_pinned() -> None:
    assert set(fe.__all__) == _EXPECTED_ALL


def test_all_names_resolve() -> None:
    # Every advertised name must actually be importable from the top level.
    for name in fe.__all__:
        assert hasattr(fe, name), f"{name} in __all__ but not importable"


def test_indexposting_is_exported() -> None:
    # IndexPosting is the return type of HashIndex.query/query_many, so it must
    # be reachable from the top-level package (not only fingerprint_engine.core).
    assert fe.IndexPosting is IndexPosting


def test_searchresult_field_order_is_frozen() -> None:
    # confidence follows score (matches to_dict() + VERSIONING §3). This layout
    # is positional and frozen at 1.0 -- reordering is a MAJOR break.
    names = [f.name for f in dataclasses.fields(SearchResult)]
    assert names == [
        "file_id",
        "score",
        "confidence",
        "aligned_votes",
        "total_votes",
        "unique_hashes",
        "offset",
        "metadata",
    ]


def test_searchresult_to_dict_key_order_matches_fields() -> None:
    result = SearchResult(
        file_id="f",
        score=1.0,
        confidence=0.5,
        aligned_votes=3,
        total_votes=4,
        unique_hashes=2,
        offset=0,
    )
    field_names = [f.name for f in dataclasses.fields(SearchResult)]
    assert list(result.to_dict().keys()) == field_names


def test_searchresult_confidence_is_required() -> None:
    # The silent confidence=0.0 default is gone: omitting it is a TypeError, so a
    # result can never be born with the always-rejected 0.0 by accident.
    import pytest

    with pytest.raises(TypeError):
        SearchResult(  # type: ignore[call-arg]
            file_id="f",
            score=1.0,
            aligned_votes=1,
            total_votes=1,
            unique_hashes=1,
            offset=0,
        )


def test_indexposting_field_order_is_frozen() -> None:
    assert [f.name for f in dataclasses.fields(IndexPosting)] == [
        "file_id",
        "hash_code",
        "time_offset",
    ]


def test_calibration_field_order_is_frozen() -> None:
    assert [f.name for f in dataclasses.fields(Calibration)] == [
        "default_min_confidence",
        "per_handler",
        "offset_tolerance",
    ]
