"""Shared test helpers: fingerprint builders, the cross-backend matrix, and the
Postgres gating, used across the (split) index test modules and elsewhere.

Imported as ``from _fixtures import ...`` -- pytest's import mode puts the tests/
directory on sys.path, so this resolves without a package. The ``pg_index``
*fixture* lives in conftest.py (fixtures auto-inject); the builders and the
``requires_pg`` marker live here because they are imported by name.
"""

from __future__ import annotations

import os

import pytest

from fingerprint_engine.core.index import (
    InMemoryHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from fingerprint_engine.core.models import ConstellationHash, Fingerprint

# Postgres integration tests need a live server; set FINGERPRINT_TEST_PG_DSN to run them.
PG_DSN = os.environ.get("FINGERPRINT_TEST_PG_DSN")
requires_pg = pytest.mark.skipif(
    not PG_DSN, reason="set FINGERPRINT_TEST_PG_DSN to run Postgres tests"
)


def fake_redis():  # noqa: ANN201 - test helper
    fakeredis = pytest.importorskip("fakeredis", exc_type=ImportError)
    return fakeredis.FakeStrictRedis(decode_responses=True)


def make_fingerprint(file_id: str, offsets: list[int]) -> Fingerprint:
    hashes = [
        ConstellationHash(
            hash_code=1000 + index,
            time_offset=offset,
            anchor_time=offset,
            target_time=offset + 1,
            freq1=10 + index,
            freq2=20 + index,
            delta_t=1,
        )
        for index, offset in enumerate(offsets)
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


def fp_with_hashes(file_id: str, code_offsets: list[tuple[int, int]]) -> Fingerprint:
    """Fingerprint with explicit (hash_code, time_offset) pairs."""
    hashes = [
        ConstellationHash(hash_code=code, time_offset=offset, anchor_time=offset,
                          target_time=offset + 1, freq1=1, freq2=2, delta_t=1)
        for code, offset in code_offsets
    ]
    return Fingerprint(file_id=file_id, path=f"/tmp/{file_id}", handler="test", size_bytes=10,
                       content_sha256=file_id, config={}, hashes=hashes, metadata={})


def search_tuples(index, query: Fingerprint, top_k: int = 10) -> list[tuple[str, int, int, float]]:  # noqa: ANN001
    """The cross-backend-comparable shape of a ranked result list.

    Pins exactly the fields whose computation is shared in the base class
    (file_id, winning offset, aligned votes, score) and whose ORDER is the
    contract every backend must reproduce byte-identically.
    """

    return [(r.file_id, r.offset, r.aligned_votes, r.score) for r in index.search(query, top_k=top_k)]


def parity_backends() -> list[tuple[str, object]]:
    """In-memory, SQLite and (if fakeredis is installed) Redis, fresh each call.

    Postgres is intentionally NOT here: it needs a live server and is gated by
    @requires_pg in the backend test module.
    """

    backends: list[tuple[str, object]] = [
        ("in_memory", InMemoryHashIndex()),
        ("sqlite", SQLiteHashIndex(":memory:")),
    ]
    try:
        import fakeredis  # noqa: F401
    except ImportError:
        pass
    else:
        backends.append(("redis", RedisHashIndex(client=fake_redis(), key_prefix="parity")))
    return backends
