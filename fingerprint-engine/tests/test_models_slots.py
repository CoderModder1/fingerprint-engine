"""Output-equivalence gate for the __slots__ memory optimisation.

These tests pin the behaviour the ``slots=True`` change MUST preserve on the
three frozen value types (IndexPosting, LandmarkPoint, ConstellationHash):

* the memory win is real — slotted instances carry NO per-instance __dict__;
* equality, hashing, ordering, and to_dict()/from_dict() round-trips are
  byte-for-byte unchanged, including use as set/dict keys (search keys on hash
  codes and the in-memory index keys postings on them);
* a measured per-instance size drop versus an equivalent un-slotted class;
* an end-to-end fingerprint -> index -> search produces unchanged hashes and
  ranked results.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import (
    ConstellationHash,
    Fingerprint,
    IndexPosting,
    LandmarkPoint,
)


def _landmark() -> LandmarkPoint:
    return LandmarkPoint(time_index=3, frequency_bin=7, magnitude=1.5)


def _hash() -> ConstellationHash:
    return ConstellationHash(
        hash_code=12345,
        time_offset=8,
        anchor_time=8,
        target_time=11,
        freq1=10,
        freq2=20,
        delta_t=3,
    )


def _posting() -> IndexPosting:
    return IndexPosting(file_id="file-a", hash_code=12345, time_offset=8)


# --- (a) the memory win: no per-instance __dict__ -------------------------------


@pytest.mark.parametrize("obj", [_landmark(), _hash(), _posting()])
def test_slotted_instances_have_no_dict(obj: object) -> None:
    # The whole point of slots=True: instances must NOT carry a __dict__.
    assert not hasattr(obj, "__dict__")


@pytest.mark.parametrize("cls", [LandmarkPoint, ConstellationHash, IndexPosting])
def test_classes_declare_slots(cls: type) -> None:
    # dataclass(slots=True) attaches a __slots__ enumerating exactly the fields.
    assert hasattr(cls, "__slots__")
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    assert tuple(cls.__slots__) == field_names


@pytest.mark.parametrize("obj", [_landmark(), _hash(), _posting()])
def test_slots_forbids_ad_hoc_attributes(obj: object) -> None:
    # A slotted frozen instance must reject any attribute outside its declared
    # fields and must NOT be mutated by the attempt. On CPython 3.13 the
    # frozen+slots class is recreated to install __slots__, so the zero-arg
    # super() in the generated __setattr__ raises TypeError (gh-90562) rather
    # than the AttributeError a plain-slots class would raise; either way the
    # write is rejected. We assert the contract that matters here: no ad-hoc
    # attribute is ever stored, which is what proves the per-instance __dict__
    # is gone (the actual memory win).
    before = repr(obj)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        obj.extra_attr = 1  # type: ignore[attr-defined]
    assert not hasattr(obj, "extra_attr")
    assert not hasattr(obj, "__dict__")
    assert repr(obj) == before


# --- (b) behaviour is unchanged -------------------------------------------------


def test_equality_unchanged() -> None:
    assert _landmark() == _landmark()
    assert _hash() == _hash()
    assert _posting() == _posting()
    assert _posting() != IndexPosting(file_id="file-b", hash_code=12345, time_offset=8)


def test_hashing_and_set_dict_key_usage_unchanged() -> None:
    # Search builds sets of hash codes and the in-memory index keys on these
    # value objects, so they must stay hashable with equal-object-equal-hash.
    for factory in (_landmark, _hash, _posting):
        a, b = factory(), factory()
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
        assert {a: "v"}[b] == "v"


def test_landmark_ordering_unchanged() -> None:
    # order=True must still derive comparisons from the field tuple in order.
    low = LandmarkPoint(time_index=1, frequency_bin=0, magnitude=9.0)
    high = LandmarkPoint(time_index=2, frequency_bin=0, magnitude=0.0)
    assert low < high
    assert sorted([high, low]) == [low, high]
    # ConstellationHash / IndexPosting are NOT order=True -> still unorderable.
    with pytest.raises(TypeError):
        _ = _hash() < _hash()


def test_to_dict_from_dict_round_trips_unchanged() -> None:
    lm = _landmark()
    assert LandmarkPoint.from_dict(lm.to_dict()) == lm
    ch = _hash()
    assert ConstellationHash.from_dict(ch.to_dict()) == ch
    assert ch.to_tuple() == (ch.hash_code, ch.time_offset)
    ip = _posting()
    assert IndexPosting.from_dict(ip.to_dict()) == ip


def test_dataclasses_asdict_still_works_on_slotted_types() -> None:
    # asdict is only called on FingerprintConfig in the codebase, but it must
    # remain functional on the slotted value types for parity with un-slotted.
    assert dataclasses.asdict(_landmark()) == _landmark().to_dict()
    assert dataclasses.asdict(_posting()) == _posting().to_dict()
    # replace() (used elsewhere on config) also still works with slots.
    moved = dataclasses.replace(_posting(), time_offset=99)
    assert moved.time_offset == 99
    assert moved.file_id == "file-a"


def test_frozen_still_enforced() -> None:
    # slots must not weaken frozen=True.
    with pytest.raises(dataclasses.FrozenInstanceError):
        _posting().time_offset = 0  # type: ignore[misc]


# --- (c) measured per-instance size drop ----------------------------------------


@dataclasses.dataclass(frozen=True)
class _UnslottedPosting:
    file_id: str
    hash_code: int
    time_offset: int


def test_measured_size_drop_versus_unslotted() -> None:
    slotted = _posting()
    unslotted = _UnslottedPosting(file_id="file-a", hash_code=12345, time_offset=8)
    # A slotted instance has no __dict__; an un-slotted one does. The honest
    # win is the eliminated __dict__, so compare total footprint including it.
    slotted_size = sys.getsizeof(slotted)
    unslotted_size = sys.getsizeof(unslotted) + sys.getsizeof(vars(unslotted))
    assert not hasattr(slotted, "__dict__")
    assert hasattr(unslotted, "__dict__")
    assert slotted_size < unslotted_size


# --- end-to-end: hashes and ranked results unchanged ----------------------------


def _make_fingerprint(file_id: str, offsets: list[int]) -> Fingerprint:
    hashes = [
        ConstellationHash(
            hash_code=1000 + i,
            time_offset=off,
            anchor_time=off,
            target_time=off + 1,
            freq1=10 + i,
            freq2=20 + i,
            delta_t=1,
        )
        for i, off in enumerate(offsets)
    ]
    return Fingerprint(
        file_id=file_id,
        path=f"/tmp/{file_id}",
        handler="test",
        size_bytes=10,
        content_sha256=file_id,
        config={},
        hashes=hashes,
        metadata={"label": file_id},
    )


def test_end_to_end_index_search_unchanged() -> None:
    index = InMemoryHashIndex()
    index.add(_make_fingerprint("aligned", [10, 20, 30, 40]))
    index.add(_make_fingerprint("scattered", [2, 40, 99, 125]))
    query = _make_fingerprint("query", [3, 13, 23, 33])

    results = index.search(query, top_k=2)

    # Same ranking contract as the canonical index test, proving slots did not
    # disturb hashing/equality used by the search vote tally.
    assert results[0].file_id == "aligned"
    assert results[0].aligned_votes == 4
    assert results[0].offset == 7

    # Postings round-trip through query() as hashable, equal value objects.
    postings = index.query(1000)
    assert IndexPosting(file_id="aligned", hash_code=1000, time_offset=10) in postings
