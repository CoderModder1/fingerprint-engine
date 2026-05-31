"""Cross-backend parity: ranking, add_many equivalence, enumeration."""

from __future__ import annotations

import threading
from pathlib import Path

from _fixtures import (
    PG_DSN,
    fp_with_hashes,
    make_fingerprint,
    requires_pg,
)
from _fixtures import (
    fake_redis as _fake_redis,
)
from _fixtures import (
    parity_backends as _parity_backends,
)
from _fixtures import (
    search_tuples as _search_tuples,
)

from fingerprint_engine.core.index import (
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from fingerprint_engine.core.models import (
    Fingerprint,
)

# ---------------------------------------------------------------------------
# Cross-backend invariant parity (Task 4)
#
# These pin deliberately-implemented invariants whose divergence point had no
# test: the offset tie-break (implemented in three places), the shared base
# scoring/ranking (must stay byte-identical across backends), and the snapshot
# interop preserving postings + metadata + ranks across a save/load between
# different backends. Redis runs via fakeredis (importorskip); Postgres parity
# is gated behind @requires_pg.
# ---------------------------------------------------------------------------




def _tie_corpus_and_query() -> tuple[Fingerprint, Fingerprint]:
    """One indexed file with TWO equal-vote offset bins, plus its query.

    The query has four distinct codes all at offset 0. Codes 1,2 sit at offset
    100 in the indexed file (delta 100, two votes); codes 3,4 sit at offset 200
    (delta 200, two votes). The offset histogram is therefore {100: 2, 200: 2}
    -- a genuine tie. The tie-break rule (votes DESC, then offset ASC) must pick
    the SMALLER offset, 100. This is the exact divergence point of the three
    implementations: in-memory ``max(..., key=lambda kv: (kv[1], -kv[0]))`` and
    the SQL ``ROW_NUMBER() ... ORDER BY votes DESC, delta ASC``.
    """

    indexed = fp_with_hashes("tie", [(1, 100), (2, 100), (3, 200), (4, 200)])
    query = fp_with_hashes("q", [(1, 0), (2, 0), (3, 0), (4, 0)])
    return indexed, query


def test_offset_tie_break_picks_smaller_offset_across_backends() -> None:
    # INVARIANT 1: winning-offset tie-break parity. Build a real tie (two offset
    # bins with equal votes) and assert every backend returns the SMALLER offset.
    # If either tie-break flipped (e.g. to offset DESC, or +kv[0]), this fails.
    indexed, query = _tie_corpus_and_query()

    winners: dict[str, tuple[int, int]] = {}
    for name, index in _parity_backends():
        index.add(indexed)
        results = index.search(query, top_k=5)
        assert results, name
        top = results[0]
        # Sanity-check the tie is real on the in-memory backend's histogram:
        # both bins must carry the same (winning) vote count.
        winners[name] = (top.offset, top.aligned_votes)

    # Every backend agrees on the smaller offset (100), with the tied vote count.
    assert set(winners.values()) == {(100, 2)}, winners

    # Belt-and-suspenders: the losing bin really has equal votes, so this
    # exercises the tie-break rather than a plain max.
    histogram: dict[int, int] = {}
    mem = InMemoryHashIndex()
    mem.add(indexed)
    for query_hash in query.hashes:
        for posting in mem.query(query_hash.hash_code):
            delta = posting.time_offset - query_hash.time_offset
            histogram[delta] = histogram.get(delta, 0) + 1
    assert histogram == {100: 2, 200: 2}  # two bins, equal votes -> genuine tie


def test_cross_backend_ranking_is_byte_identical() -> None:
    # INVARIANT 2: shared base scoring/ranking is byte-identical across backends.
    # Index the same corpus into every backend and assert search() returns the
    # SAME (file_id, offset, aligned_votes, score) ordering for one query.
    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)])

    per_backend: dict[str, list[tuple[str, int, int, float]]] = {}
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        per_backend[name] = _search_tuples(index, query, top_k=10)

    # A meaningful ranking (more than one match) actually ordered, not trivial.
    reference = per_backend["in_memory"]
    assert len(reference) >= 3
    assert reference[0][0] == "alpha"  # the coherently-aligned file ranks first
    for name, tuples in per_backend.items():
        assert tuples == reference, (name, tuples, reference)


def test_snapshot_interop_preserves_postings_metadata_and_ranks() -> None:
    # INVARIANT 3: a snapshot save()'d by one backend, load_snapshot'd into
    # another, preserves postings + metadata + ranks identically. Existing tests
    # only check file_count/posting_count + the top result; this pins the full
    # postings set, the metadata map, and the exact ranked ordering.
    import tempfile
    from pathlib import Path as _Path

    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)])

    def normalized_files(index) -> dict[str, list[tuple[int, int]]]:
        # to_dict() preserves per-file order differently per backend, so compare
        # the postings as sorted multisets -- equality of CONTENT, not order.
        return {
            file_id: sorted((int(code), int(offset)) for code, offset in entries)
            for file_id, entries in index.to_dict()["files"].items()
        }

    def metadata_map(index) -> dict[str, dict]:
        return {file_id: index._metadata_for(file_id) for file_id in index.to_dict()["files"]}

    # Save once from SQLite (a SQL backend, so it round-trips the signed-offset
    # encode), then load into every backend and compare against the source.
    source = SQLiteHashIndex(":memory:")
    for fingerprint in corpus:
        source.add(fingerprint)
    source_files = normalized_files(source)
    source_meta = metadata_map(source)
    source_ranks = _search_tuples(source, query, top_k=10)

    with tempfile.TemporaryDirectory() as directory:
        snapshot = _Path(directory) / "snap.json"
        source.save(snapshot)

        targets: list[tuple[str, object]] = [
            ("in_memory", InMemoryHashIndex().load_snapshot(snapshot)),
            ("sqlite", SQLiteHashIndex(":memory:").load_snapshot(snapshot)),
        ]
        try:
            import fakeredis  # noqa: F401
        except ImportError:
            pass
        else:
            targets.append(
                ("redis", RedisHashIndex(client=_fake_redis(), key_prefix="snap").load_snapshot(snapshot))
            )
        # The classmethod load() entry point must agree with load_snapshot() too.
        targets.append(("in_memory_load", InMemoryHashIndex.load(snapshot)))

        for name, target in targets:
            assert normalized_files(target) == source_files, name
            assert metadata_map(target) == source_meta, name
            assert _search_tuples(target, query, top_k=10) == source_ranks, name


@requires_pg
def test_postgres_offset_tie_break_picks_smaller_offset(pg_index) -> None:
    # INVARIANT 1 for Postgres: its ROW_NUMBER() ... ORDER BY votes DESC, delta
    # ASC must resolve a real tie to the smaller offset, like the others.
    indexed, query = _tie_corpus_and_query()
    pg_index.add(indexed)

    top = pg_index.search(query, top_k=5)[0]
    assert (top.offset, top.aligned_votes) == (100, 2)


@requires_pg
def test_postgres_ranking_matches_in_memory_byte_identical(pg_index) -> None:
    # INVARIANT 2 for Postgres: full (file_id, offset, aligned_votes, score)
    # ordering identical to the in-memory reference.
    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)])

    mem = InMemoryHashIndex()
    for fingerprint in corpus:
        mem.add(fingerprint)
        pg_index.add(fingerprint)

    assert _search_tuples(pg_index, query, top_k=10) == _search_tuples(mem, query, top_k=10)


@requires_pg
def test_postgres_candidate_limit_matches_in_memory(pg_index) -> None:
    # A6 for Postgres: the shared-POSTING candidate prefilter (base-class
    # _select_candidates over PG's query_many) must keep a repeated-code high-vote
    # match under a tight limit AND rank identically to in-memory. _parity_backends
    # excludes PG, so this is where PG candidate_limit parity is proven.
    coherent = [(1, off) for off in range(0, 1000, 50)]
    query = fp_with_hashes(
        "q", coherent + [(2, 5), (3, 6), (10, 7), (11, 8), (12, 9), (13, 10), (14, 11)]
    )
    true_file = fp_with_hashes(
        "true", [(1, off + 100) for off in range(0, 1000, 50)] + [(2, 500), (3, 600)]
    )
    decoys = [
        fp_with_hashes(f"decoy{i}", [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0)]) for i in range(8)
    ]
    mem = InMemoryHashIndex()
    for fingerprint in [true_file, *decoys]:
        mem.add(fingerprint)
        pg_index.add(fingerprint)

    def tuples(index, limit):  # noqa: ANN001, ANN202 - test helper
        return [
            (r.file_id, r.offset, r.aligned_votes, r.score)
            for r in index.search(query, top_k=10, candidate_limit=limit)
        ]

    for limit in (5, 100, None):
        assert tuples(pg_index, limit) == tuples(mem, limit), limit
    top = pg_index.search(query, top_k=1, candidate_limit=5)
    assert top and top[0].file_id == "true"


@requires_pg
def test_postgres_concurrent_add_and_search_stay_consistent(pg_index) -> None:
    # The @_synchronized per-index lock must keep concurrent add()/search() on one
    # shared Postgres connection (as the FastAPI threadpool issues) consistent --
    # no InterfaceError, no torn multi-statement section, no miscounted postings.
    file_ids = [f"file-{i}" for i in range(20)]
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(file_ids))

    def worker(file_id: str) -> None:
        try:
            barrier.wait()
            pg_index.add(make_fingerprint(file_id, [1, 2, 3]))
            pg_index.search(make_fingerprint("query", [1, 2, 3]), top_k=5)
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(fid,)) for fid in file_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert pg_index.file_count == len(file_ids)
    assert pg_index.posting_count == 3 * len(file_ids)


@requires_pg
def test_postgres_snapshot_interop_preserves_postings_metadata_and_ranks(pg_index, tmp_path: Path) -> None:
    # INVARIANT 3 for Postgres: a Postgres-written snapshot loads into the
    # in-memory backend with identical postings, metadata, and ranks.
    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1)])
    for fingerprint in corpus:
        pg_index.add(fingerprint)

    def normalized_files(index) -> dict[str, list[tuple[int, int]]]:
        return {
            file_id: sorted((int(code), int(offset)) for code, offset in entries)
            for file_id, entries in index.to_dict()["files"].items()
        }

    snapshot = tmp_path / "snap.json"
    pg_index.save(snapshot)
    loaded = InMemoryHashIndex.load(snapshot)

    assert normalized_files(loaded) == normalized_files(pg_index)
    assert {fid: loaded._metadata_for(fid) for fid in normalized_files(loaded)} == {
        fid: pg_index._metadata_for(fid) for fid in normalized_files(pg_index)
    }
    assert _search_tuples(loaded, query, top_k=10) == _search_tuples(pg_index, query, top_k=10)


# ---------------------------------------------------------------------------
# Bulk/transactional ingest parity (Task 3 -- add_many)
#
# add_many() MUST be exactly equivalent to calling add() per fingerprint in
# sequence: same posting_count, same normalized postings + metadata snapshot,
# and same search() rankings -- including the replace case where a file_id is
# re-indexed. These pin that contract for InMemory, SQLite, and (via fakeredis)
# Redis. Postgres uses COPY and is gated behind @requires_pg below.
# ---------------------------------------------------------------------------


def _bulk_parity_factories() -> list[tuple[str, object]]:
    """One factory per available backend: ``factory()`` returns a FRESH index.

    Each parity test needs two independent indexes (sequential vs add_many) and
    several distinct-prefixed Redis namespaces, so a factory (not a single
    pre-built index) keeps them isolated.
    """

    counter = {"n": 0}

    def redis_factory():
        counter["n"] += 1
        return RedisHashIndex(client=_fake_redis(), key_prefix=f"bulk{counter['n']}")

    factories: list[tuple[str, object]] = [
        ("in_memory", InMemoryHashIndex),
        ("sqlite", lambda: SQLiteHashIndex(":memory:")),
    ]
    try:
        import fakeredis  # noqa: F401
    except ImportError:
        pass
    else:
        factories.append(("redis", redis_factory))
    return factories


def _normalized_files(index) -> dict[str, list[tuple[int, int]]]:
    # Compare postings as sorted multisets: to_dict() per-file order differs by
    # backend, but the CONTENT must be identical.
    return {
        file_id: sorted((int(code), int(offset)) for code, offset in entries)
        for file_id, entries in index.to_dict()["files"].items()
    }


def _metadata_map(index) -> dict[str, dict]:
    return {file_id: index._metadata_for(file_id) for file_id in index.to_dict()["files"]}


def _assert_bulk_equivalent(name, sequential, bulk, queries) -> None:
    # The core parity gate: add_many() == sequential add() for every observable.
    assert bulk.file_count == sequential.file_count, name
    assert bulk.posting_count == sequential.posting_count, name
    assert _normalized_files(bulk) == _normalized_files(sequential), name
    assert _metadata_map(bulk) == _metadata_map(sequential), name
    for query in queries:
        assert _search_tuples(bulk, query, top_k=20) == _search_tuples(sequential, query, top_k=20), (
            name, query.file_id,
        )


def _bulk_corpus() -> list[Fingerprint]:
    return [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
        # A file with zero hashes: add()/remove() must still track membership.
        fp_with_hashes("empty", []),
    ]


def _bulk_queries() -> list[Fingerprint]:
    return [
        fp_with_hashes("q1", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)]),
        fp_with_hashes("q2", [(1, 0), (2, 0), (3, 0)]),
    ]


def test_add_many_equals_sequential_add_across_backends() -> None:
    corpus = _bulk_corpus()
    queries = _bulk_queries()
    for name, factory in _bulk_parity_factories():
        sequential = factory()
        for fingerprint in corpus:
            sequential.add(fingerprint)
        bulk = factory()
        bulk.add_many(corpus)
        _assert_bulk_equivalent(name, sequential, bulk, queries)


def test_add_many_preserves_replace_semantics_across_backends() -> None:
    # A batch containing a file_id already present (and a duplicate WITHIN the
    # batch) must end in the same state as sequential add(): last writer wins,
    # pre-existing postings removed exactly once.
    queries = _bulk_queries()
    # Re-index list: alpha replaced (different postings), beta new, alpha AGAIN
    # later in the same batch (intra-batch duplicate -> the final alpha wins).
    reindex = [
        fp_with_hashes("alpha", [(7, 1), (8, 2)]),
        fp_with_hashes("beta", [(1, 5), (2, 40)]),
        fp_with_hashes("alpha", [(9, 3), (10, 4), (11, 5)]),  # later duplicate wins
    ]
    for name, factory in _bulk_parity_factories():
        sequential = factory()
        # Pre-populate so add_many must REPLACE an existing alpha.
        for fingerprint in [fp_with_hashes("alpha", [(1, 100), (2, 200)]), fp_with_hashes("gamma", [(3, 1)])]:
            sequential.add(fingerprint)
        bulk = factory()
        for fingerprint in [fp_with_hashes("alpha", [(1, 100), (2, 200)]), fp_with_hashes("gamma", [(3, 1)])]:
            bulk.add(fingerprint)

        for fingerprint in reindex:
            sequential.add(fingerprint)  # sequential reference path
        bulk.add_many(reindex)           # bulk path under test

        _assert_bulk_equivalent(name, sequential, bulk, queries)
        # Concrete replace check: the later alpha (codes 9,10,11) is the survivor,
        # the pre-batch alpha (codes 1,2) is gone -- on both paths.
        assert _normalized_files(bulk)["alpha"] == [(9, 3), (10, 4), (11, 5)], name


def test_add_many_empty_iterable_is_noop_across_backends() -> None:
    for name, factory in _bulk_parity_factories():
        index = factory()
        index.add_many([])
        assert index.file_count == 0, name
        assert index.posting_count == 0, name


def test_sqlite_add_many_uses_single_commit() -> None:
    # The whole point of the SQLite override: ONE commit for the batch, not one
    # per file. We prove the commit-count drop by counting sqlite3 commits via a
    # wrapped connection, and (separately) that synchronous=NORMAL is set.
    import sqlite3 as _sqlite3

    commits = {"n": 0}
    real_connect = _sqlite3.connect

    class CountingConnection:
        def __init__(self, conn):
            self._conn = conn

        def commit(self):
            commits["n"] += 1
            return self._conn.commit()

        def __getattr__(self, attr):
            return getattr(self._conn, attr)

    raw = real_connect(":memory:", check_same_thread=False)
    index = SQLiteHashIndex(connection=CountingConnection(raw))

    # synchronous=NORMAL (value 1) is set by _init_schema for the WAL pairing.
    assert int(raw.execute("PRAGMA synchronous").fetchone()[0]) == 1

    corpus = [make_fingerprint(f"f{i}", [1, 2, 3, 4]) for i in range(25)]
    commits["n"] = 0  # ignore schema-setup commits; count only the ingest
    index.add_many(corpus)

    # add() would commit at least once per file (>=25); add_many commits ONCE.
    assert commits["n"] == 1
    assert index.file_count == 25
    assert index.posting_count == 100


@requires_pg
def test_pg_add_many_equals_sequential_add(pg_index) -> None:
    # Postgres parity: COPY-based add_many() must match sequential add(). Uses a
    # second prefix for the sequential reference so the two never share tables.
    corpus = _bulk_corpus()
    queries = _bulk_queries()

    sequential = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest_seq")
    with sequential._conn.cursor() as cur:
        cur.execute(f"TRUNCATE {sequential._files_table}, {sequential._postings_table}")
    sequential._conn.commit()
    try:
        for fingerprint in corpus:
            sequential.add(fingerprint)
        pg_index.add_many(corpus)
        _assert_bulk_equivalent("postgres", sequential, pg_index, queries)
    finally:
        with sequential._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {sequential._postings_table}, {sequential._files_table}")
        sequential._conn.commit()
        sequential.close()


# ---------------------------------------------------------------------------
# Enumeration API parity (Task 1 -- list_files / iter_metadata / contains)
#
# list_files() and iter_metadata() MUST be byte-identical across every backend
# and reflect adds AND removes; iter_metadata() MUST yield exactly what
# _metadata_for() returns, in list_files() order, without going through the
# heavy to_dict(). contains()/__contains__ MUST agree with list_files()
# membership. Redis runs via fakeredis; Postgres parity is gated by @requires_pg.
# ---------------------------------------------------------------------------


def test_list_files_and_iter_metadata_parity_across_backends() -> None:
    corpus = _bulk_corpus()  # includes an "empty" (zero-hash) file: still listed
    expected_ids = sorted(fingerprint.file_id for fingerprint in corpus)

    per_backend_ids: dict[str, list[str]] = {}
    per_backend_meta: dict[str, list[dict]] = {}
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)

        listed = index.list_files()
        # Sorted, deterministic, and covers every added file (empty one included).
        assert listed == expected_ids, name
        per_backend_ids[name] = listed

        # iter_metadata() must yield exactly _metadata_for() for each id, IN ORDER,
        # and not depend on to_dict() (it streams list_files()+_metadata_for()).
        streamed = list(index.iter_metadata())
        assert [m["file_id"] for m in streamed] == expected_ids, name
        assert streamed == [index._metadata_for(fid) for fid in listed], name
        per_backend_meta[name] = streamed

        # Membership: contains()/__contains__ agree with list_files().
        for fid in expected_ids:
            assert index.contains(fid), (name, fid)
            assert fid in index, (name, fid)
        assert not index.contains("does-not-exist"), name
        assert "does-not-exist" not in index, name
        assert 12345 not in index, name  # non-str membership is False, not a crash

    # Cross-backend byte-identical enumeration (ids and full metadata dicts).
    reference_ids = per_backend_ids["in_memory"]
    reference_meta = per_backend_meta["in_memory"]
    for name in per_backend_ids:
        assert per_backend_ids[name] == reference_ids, name
        assert per_backend_meta[name] == reference_meta, name


def test_list_files_reflects_adds_and_removes_across_backends() -> None:
    for name, index in _parity_backends():
        assert index.list_files() == [], name  # empty index enumerates to nothing
        assert list(index.iter_metadata()) == [], name

        index.add(fp_with_hashes("alpha", [(1, 10), (2, 20)]))
        index.add(fp_with_hashes("beta", [(3, 5)]))
        assert index.list_files() == ["alpha", "beta"], name
        assert "alpha" in index and "beta" in index, name

        index.remove("alpha")
        assert index.list_files() == ["beta"], name
        assert "alpha" not in index, name
        assert [m["file_id"] for m in index.iter_metadata()] == ["beta"], name

        # Re-adding (replace) a file_id keeps it listed exactly once.
        index.add(fp_with_hashes("beta", [(9, 1), (9, 2)]))
        assert index.list_files() == ["beta"], name
        assert index._metadata_for("beta")["hash_count"] == 2, name


def test_iter_metadata_does_not_call_to_dict() -> None:
    # iter_metadata() must stream list_files()+_metadata_for(), NOT build the heavy
    # whole-index to_dict(). Trip a flag if to_dict() is touched during iteration.
    index = InMemoryHashIndex()
    for fingerprint in _bulk_corpus():
        index.add(fingerprint)

    called = {"to_dict": False}
    original = index.to_dict

    def _tripwire() -> dict:
        called["to_dict"] = True
        return original()

    index.to_dict = _tripwire  # type: ignore[method-assign]
    streamed = list(index.iter_metadata())

    assert not called["to_dict"]
    assert [m["file_id"] for m in streamed] == sorted(fp.file_id for fp in _bulk_corpus())


def test_list_files_and_iter_metadata_survive_snapshot_round_trip(tmp_path: Path) -> None:
    # Enumeration must reflect a load_snapshot() bulk import, not just live adds.
    corpus = _bulk_corpus()
    source = SQLiteHashIndex(":memory:")
    for fingerprint in corpus:
        source.add(fingerprint)
    snapshot = tmp_path / "snap.json"
    source.save(snapshot)

    loaded = InMemoryHashIndex.load(snapshot)
    assert loaded.list_files() == source.list_files()
    assert list(loaded.iter_metadata()) == list(source.iter_metadata())


@requires_pg
def test_pg_list_files_and_iter_metadata_match_in_memory(pg_index) -> None:
    # Enumeration parity for Postgres against the in-memory reference, including
    # the empty (zero-hash) file and removal.
    corpus = _bulk_corpus()
    mem = InMemoryHashIndex()
    for fingerprint in corpus:
        mem.add(fingerprint)
        pg_index.add(fingerprint)

    assert pg_index.list_files() == mem.list_files()
    assert list(pg_index.iter_metadata()) == list(mem.iter_metadata())
    for fid in mem.list_files():
        assert pg_index.contains(fid)
        assert fid in pg_index
    assert "missing" not in pg_index

    pg_index.remove("alpha")
    mem.remove("alpha")
    assert pg_index.list_files() == mem.list_files()
    assert "alpha" not in pg_index


@requires_pg
def test_pg_add_many_preserves_replace_semantics(pg_index) -> None:
    queries = _bulk_queries()
    pg_index.add(fp_with_hashes("alpha", [(1, 100), (2, 200)]))
    pg_index.add(fp_with_hashes("gamma", [(3, 1)]))

    reference = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest_seq")
    with reference._conn.cursor() as cur:
        cur.execute(f"TRUNCATE {reference._files_table}, {reference._postings_table}")
    reference._conn.commit()
    try:
        reference.add(fp_with_hashes("alpha", [(1, 100), (2, 200)]))
        reference.add(fp_with_hashes("gamma", [(3, 1)]))
        reindex = [
            fp_with_hashes("alpha", [(7, 1), (8, 2)]),
            fp_with_hashes("beta", [(1, 5), (2, 40)]),
            fp_with_hashes("alpha", [(9, 3), (10, 4), (11, 5)]),
        ]
        for fingerprint in reindex:
            reference.add(fingerprint)
        pg_index.add_many(reindex)
        _assert_bulk_equivalent("postgres", reference, pg_index, queries)
        assert _normalized_files(pg_index)["alpha"] == [(9, 3), (10, 4), (11, 5)]
    finally:
        with reference._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {reference._postings_table}, {reference._files_table}")
        reference._conn.commit()
        reference.close()


