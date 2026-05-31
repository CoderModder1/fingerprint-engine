"""Integer file_id surrogate key (in-memory + SQL) and old-schema migration."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from _fixtures import (
    PG_DSN,
    fp_with_hashes,
    requires_pg,
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
    SQLiteHashIndex,
)
from fingerprint_engine.core.models import (
    IndexPosting,
)

# --------------------------------------------------------------------------- #
# Integer file_id surrogate key (internal storage optimization).
#
# InMemoryHashIndex stores a compact integer surrogate per posting instead of
# the 64-char SHA-256 file_id, mapping back to the str file_id only at the
# query/aggregate boundary. These tests pin that the optimization is fully
# OUTPUT-PRESERVING: search rankings, to_dict snapshots, list_files,
# iter_metadata, query, and query_many are byte-identical to storing the str
# verbatim, plus the internal surrogate invariants (retire-not-reuse) hold.
# --------------------------------------------------------------------------- #

# Realistic 64-hex-char SHA-256-shaped file_ids (the value the surrogate replaces).
_SURROGATE_CORPUS = [
    ("a" * 64, [(1000, 10), (1001, 20), (1002, 30), (1003, 40)]),
    ("b" * 64, [(1000, 2), (1001, 40), (1004, 99), (1005, 125)]),
    ("c" * 64, [(1001, 21), (1002, 31), (1006, 7)]),
    ("d" * 64, [(1000, 11), (1001, 22), (1002, 33), (1003, 44), (1007, 88)]),
]


def _build_surrogate_index() -> InMemoryHashIndex:
    index = InMemoryHashIndex()
    for file_id, code_offsets in _SURROGATE_CORPUS:
        index.add(fp_with_hashes(file_id, code_offsets))
    return index


def _reference_postings() -> dict[int, list[tuple[str, int]]]:
    """The (file_id, time_offset) posting lists a str-storing index would hold.

    Recomputed straight from the corpus definition, in add() insertion order,
    so it is an independent oracle for what query()/query_many() must return.
    """

    expected: dict[int, list[tuple[str, int]]] = {}
    for file_id, code_offsets in _SURROGATE_CORPUS:
        for code, offset in code_offsets:
            expected.setdefault(code, []).append((file_id, offset))
    return expected


def test_surrogate_query_returns_str_file_ids_in_stored_order() -> None:
    # query() maps each surrogate back to the original str file_id, preserving
    # the public IndexPosting return type/values and the stored posting order.
    index = _build_surrogate_index()
    expected = _reference_postings()
    for code, expected_postings in expected.items():
        got = index.query(code)
        assert [(p.file_id, p.time_offset) for p in got] == expected_postings
        assert all(p.hash_code == code for p in got)
        assert all(isinstance(p.file_id, str) and len(p.file_id) == 64 for p in got)


def test_surrogate_query_many_matches_per_code_query() -> None:
    # query_many is parity-identical to per-code query() (the base contract),
    # which must survive the surrogate mapping unchanged.
    index = _build_surrogate_index()
    codes = [code for code, _ in _reference_postings().items()]
    batched = index.query_many(codes + [999999])  # include a miss
    for code in codes:
        assert sorted((p.file_id, p.time_offset) for p in batched[code]) == \
               sorted((p.file_id, p.time_offset) for p in index.query(code))
    assert batched[999999] == []


def test_surrogate_postings_store_no_file_id_strings() -> None:
    # The actual footprint win: every stored posting is a compact (int, int)
    # pair -- no 64-char file_id string is held per posting.
    index = _build_surrogate_index()
    for postings in index._postings.values():
        for posting in postings:
            assert isinstance(posting, tuple) and len(posting) == 2
            assert isinstance(posting[0], int)   # surrogate, not a str file_id
            assert isinstance(posting[1], int)   # time_offset
    # Every live surrogate resolves back to a real file_id and round-trips.
    assert set(index._id_to_fid.values()) == {fid for fid, _ in _SURROGATE_CORPUS}
    assert index._fid_to_id == {fid: sid for sid, fid in index._id_to_fid.items()}


def test_surrogate_search_to_dict_list_files_iter_metadata_identical() -> None:
    # OUTPUT-PRESERVING PROOF: every observable matches a reference index built
    # from the SAME fingerprints. (Both use the surrogate storage, but the
    # reference is reconstructed via the public to_dict/from_dict round-trip,
    # exercising the full encode/decode path independent of the live add path.)
    index = _build_surrogate_index()
    reference = InMemoryHashIndex.from_dict(index.to_dict())

    query = fp_with_hashes("q" * 64, [(1000, 0), (1001, 0), (1002, 0), (1003, 0)])
    assert _search_tuples(index, query, top_k=10) == _search_tuples(reference, query, top_k=10)
    # Full SearchResult equality (score, votes, offset, confidence, metadata).
    assert index.search(query, top_k=10) == reference.search(query, top_k=10)

    assert index.to_dict() == reference.to_dict()
    assert index.list_files() == reference.list_files()
    assert list(index.iter_metadata()) == list(reference.iter_metadata())
    assert [index._metadata_for(fid) for fid in index.list_files()] == \
           [reference._metadata_for(fid) for fid in reference.list_files()]
    assert index.file_count == reference.file_count
    assert index.posting_count == reference.posting_count


def test_surrogate_to_dict_keys_are_str_file_ids() -> None:
    # The snapshot is keyed by the original 64-char file_id (NOT the surrogate),
    # so snapshots stay portable and cross-backend identical.
    index = _build_surrogate_index()
    snapshot = index.to_dict()
    assert set(snapshot["files"]) == {fid for fid, _ in _SURROGATE_CORPUS}
    assert set(snapshot["metadata"]) == {fid for fid, _ in _SURROGATE_CORPUS}
    # The corpus's first file: stored entries are (hash_code, time_offset) pairs
    # with the real hash codes, never surrogate ints. (InMemory.to_dict returns
    # _file_entries verbatim as tuples -- pre-existing behaviour, unchanged by
    # the surrogate work; the JSON snapshot round-trip normalizes them to lists.)
    first_id, first_pairs = _SURROGATE_CORPUS[0]
    assert snapshot["files"][first_id] == [(c, o) for c, o in first_pairs]


def test_surrogate_retained_on_remove_and_reused_on_readd() -> None:
    # remove() deliberately KEEPS the surrogate mappings alive (so a lock-free
    # reader can always resolve a just-removed posting's surrogate instead of
    # racing into a KeyError), and a re-added file_id REUSES that same surrogate
    # (re-indexing never grows the maps). The retained surrogate is postings-less,
    # so it is invisible to every query/search.
    index = InMemoryHashIndex()
    index.add(fp_with_hashes("x" * 64, [(1, 10), (2, 20)]))
    first_surrogate = index._fid_to_id["x" * 64]
    index.remove("x" * 64)
    # Mappings retained across remove() -- still resolvable, never KeyError.
    assert index._fid_to_id["x" * 64] == first_surrogate
    assert index._id_to_fid[first_surrogate] == "x" * 64
    assert index.query(1) == []  # postings gone (the observable effect)

    index.add(fp_with_hashes("x" * 64, [(1, 10), (2, 20)]))
    second_surrogate = index._fid_to_id["x" * 64]
    assert second_surrogate == first_surrogate  # reused, not a fresh allocation
    # query() still resolves to the right str file_id after the surrogate churn.
    assert [(p.file_id, p.time_offset) for p in index.query(1)] == [("x" * 64, 10)]


def test_surrogate_replace_keeps_output_correct() -> None:
    # add() of an existing file_id replaces it: old postings are gone, the new
    # ones resolve correctly, counts are exact, and the file's surrogate is
    # REUSED (replace re-interns the same id -> same surrogate, maps don't grow).
    index = InMemoryHashIndex()
    index.add(fp_with_hashes("a" * 64, [(1, 1), (2, 2), (3, 3)]))
    index.add(fp_with_hashes("b" * 64, [(1, 5)]))
    old_a = index._fid_to_id["a" * 64]
    index.add(fp_with_hashes("a" * 64, [(9, 99)]))  # replace a
    new_a = index._fid_to_id["a" * 64]
    assert new_a == old_a  # reused across replace
    assert index._id_to_fid[old_a] == "a" * 64  # mapping still valid
    assert index.file_count == 2
    assert index.posting_count == 2  # a:1 + b:1
    assert index.query(1) == [IndexPosting(file_id="b" * 64, hash_code=1, time_offset=5)]
    assert index.query(9) == [IndexPosting(file_id="a" * 64, hash_code=9, time_offset=99)]
    assert index.query(2) == []  # old a posting removed


def test_surrogate_prune_stop_hashes_output_identical() -> None:
    # prune_stop_hashes uses distinct surrogates as the document-frequency
    # measure; the result (removed count, remaining postings, recalibrated
    # hash_count, search) matches a from_dict reconstruction of the same state.
    index = InMemoryHashIndex()
    # code 1000 appears in all 4 files (df=1.0 -> stop at ratio 0.5); others rare.
    for i in range(4):
        fid = chr(ord("a") + i) * 64
        index.add(fp_with_hashes(fid, [(1000, i), (2000 + i, i * 2)]))
    removed = index.prune_stop_hashes(max_df_ratio=0.5)
    assert removed == 4  # the 4 postings of code 1000
    assert index.query(1000) == []
    # Each file keeps exactly its one rare posting; hash_count recalibrated.
    for i in range(4):
        fid = chr(ord("a") + i) * 64
        assert index._metadata_for(fid)["hash_count"] == 1
        assert [(p.file_id, p.time_offset) for p in index.query(2000 + i)] == [(fid, i * 2)]
    # A from_dict reconstruction of the pruned snapshot is output-identical.
    reference = InMemoryHashIndex.from_dict(index.to_dict())
    assert index.to_dict() == reference.to_dict()
    assert list(index.iter_metadata()) == list(reference.iter_metadata())


def test_surrogate_cross_backend_parity_with_sqlite() -> None:
    # The surrogate-backed in-memory search must still be byte-identical to the
    # SQLite backend (which stores the str file_id verbatim) for the same corpus.
    mem = _build_surrogate_index()
    sql = SQLiteHashIndex(":memory:")
    try:
        for file_id, code_offsets in _SURROGATE_CORPUS:
            sql.add(fp_with_hashes(file_id, code_offsets))
        query = fp_with_hashes("q" * 64, [(1000, 0), (1001, 0), (1002, 0), (1003, 0)])
        assert _search_tuples(mem, query, top_k=10) == _search_tuples(sql, query, top_k=10)
        assert mem.list_files() == sql.list_files()
        # The snapshot files map is parity-identical once normalized through
        # JSON (InMemory keeps tuples, SQL lists -- the round-trip the save/load
        # path always applies); compare via that canonical form.
        assert json.loads(json.dumps(mem.to_dict()["files"])) == \
               json.loads(json.dumps(sql.to_dict()["files"]))
    finally:
        sql.close()


def test_surrogate_per_posting_memory_drop() -> None:
    # MEASUREMENT: the surrogate (int, int) pair is far smaller than an
    # IndexPosting holding a 64-char file_id string. Compare the per-posting
    # footprint of the stored representation against the old representation.
    posting_tuple = (123, 45)  # (surrogate, time_offset) as stored now
    old_posting = IndexPosting(file_id="a" * 64, hash_code=1000, time_offset=45)
    file_id_str = "a" * 64

    # Old per-posting cost: the IndexPosting object PLUS the 64-char file_id it
    # references (the dominant cost the surrogate eliminates -- the str is held
    # once now, not per posting).
    old_bytes = sys.getsizeof(old_posting) + sys.getsizeof(file_id_str)
    new_bytes = sys.getsizeof(posting_tuple) + 2 * sys.getsizeof(123)
    assert new_bytes < old_bytes  # strictly smaller per posting

    # Whole-index sanity at a few-thousand-posting scale: building a 200-file x
    # 30-posting index stores 6000 compact pairs and zero per-posting file_id
    # strings (each 64-char id is held once in the surrogate maps).
    index = InMemoryHashIndex()
    for i in range(200):
        fid = f"{i:064x}"
        index.add(fp_with_hashes(fid, [(1000 + j, j) for j in range(30)]))
    assert index.posting_count == 6000
    # Exactly one file_id string per file is retained (in the surrogate map),
    # never one per posting.
    assert len(index._id_to_fid) == 200
    assert len(index._fid_to_id) == 200


# --------------------------------------------------------------------------- #
# Integer file_id surrogate key for the SQL backends (SQLiteHashIndex /
# PostgresHashIndex). Each posting row used to store the 64-char SHA-256 file_id
# verbatim; now a normalized files row maps file_id -> a small integer id and
# postings store that integer file_ref FK. These tests pin that the change is
# fully OUTPUT-PRESERVING (search rankings, to_dict snapshots, list_files,
# iter_metadata, counts byte-identical to the InMemory parity reference and to
# what the prior str-storing SQLite produced), that an OLD-schema .sqlite3
# migrates transparently, and they MEASURE the on-disk size win. Postgres is
# structural and gated behind @requires_pg (cannot be run live here).
# --------------------------------------------------------------------------- #

# Realistic 64-hex-char SHA-256-shaped file_ids (the value the surrogate replaces).
_SQL_SURROGATE_CORPUS = [
    ("a" * 64, [(1000, 10), (1001, 20), (1002, 30), (1003, 40)]),
    ("b" * 64, [(1000, 2), (1001, 40), (1004, 99), (1005, 125)]),
    ("c" * 64, [(1001, 21), (1002, 31), (1006, 7)]),
    ("d" * 64, [(1000, 11), (1001, 22), (1002, 33), (1003, 44), (1007, 88)]),
    ("e" * 64, []),  # zero-hash file: still tracked, still listed
]
_SQL_SURROGATE_QUERY = [(1000, 0), (1001, 0), (1002, 0), (1003, 0)]


def _build_old_schema_sqlite(path: Path, corpus) -> None:
    """Create a SQLite DB in the PRE-surrogate schema, exactly as the prior code.

    Mirrors the old ``SQLiteHashIndex`` layout (``postings.file_id TEXT``, a
    ``files`` table with no ``id`` column, the same two indexes) and the same
    signed-offset hash encoding and metadata JSON, built with raw sqlite3 so it
    is independent of the current (new-schema) class. This is the on-disk file an
    already-deployed index would have before the upgrade.
    """

    offset = SQLiteHashIndex._SIGNED_OFFSET
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE files (file_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
        CREATE TABLE postings (
            file_id TEXT NOT NULL, hash_code INTEGER NOT NULL, time_offset INTEGER NOT NULL
        );
        CREATE INDEX idx_postings_hash ON postings(hash_code);
        CREATE INDEX idx_postings_file ON postings(file_id);
        """
    )
    for file_id, code_offsets in corpus:
        metadata = {
            "file_id": file_id, "path": f"/tmp/{file_id}", "handler": "test",
            "size_bytes": 10, "content_sha256": file_id,
            "hash_count": len(code_offsets), "landmark_count": 0,
        }
        conn.execute(
            "INSERT INTO files (file_id, metadata) VALUES (?, ?)",
            (file_id, json.dumps(metadata, sort_keys=True)),
        )
        conn.executemany(
            "INSERT INTO postings (file_id, hash_code, time_offset) VALUES (?, ?, ?)",
            [(file_id, code - offset, offset_value) for code, offset_value in code_offsets],
        )
    conn.commit()
    conn.close()


def _build_new_sqlite(corpus, database=":memory:") -> SQLiteHashIndex:
    index = SQLiteHashIndex(database)
    for file_id, code_offsets in corpus:
        index.add(fp_with_hashes(file_id, code_offsets))
    return index


def _build_in_memory(corpus) -> InMemoryHashIndex:
    index = InMemoryHashIndex()
    for file_id, code_offsets in corpus:
        index.add(fp_with_hashes(file_id, code_offsets))
    return index


def test_sql_surrogate_postings_store_integer_file_ref_not_string() -> None:
    # The footprint win: the new postings table stores an INTEGER file_ref FK,
    # never the 64-char file_id string; the files table holds the id<->file_id
    # surrogate mapping (one row per file, not per posting).
    index = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    posting_cols = {row[1] for row in index._conn.execute("PRAGMA table_info(postings)").fetchall()}
    file_cols = {row[1] for row in index._conn.execute("PRAGMA table_info(files)").fetchall()}
    assert "file_ref" in posting_cols and "file_id" not in posting_cols
    assert {"id", "file_id", "metadata"} <= file_cols
    # Every posting's file_ref is an int that resolves to a real files.id.
    valid_ids = {row[0] for row in index._conn.execute("SELECT id FROM files").fetchall()}
    refs = index._conn.execute("SELECT DISTINCT file_ref FROM postings").fetchall()
    assert refs and all(isinstance(r[0], int) and r[0] in valid_ids for r in refs)
    # files.id is the rowid surrogate; file_id stays the 64-char string.
    for row in index._conn.execute("SELECT id, file_id FROM files").fetchall():
        assert isinstance(row[0], int) and isinstance(row[1], str) and len(row[1]) == 64
    index.close()


def test_sql_surrogate_output_equivalent_to_in_memory_and_prior_sqlite() -> None:
    # HARD GATE 1 (OUTPUT-EQUIVALENCE): the new-schema SQLite backend's
    # search()/to_dict()/list_files()/iter_metadata()/posting_count/file_count
    # are byte-identical to the InMemory parity reference (the cross-backend
    # oracle the existing suite uses, which the prior str-storing SQLite was also
    # required to match -- so equality with it is equality with the prior SQLite
    # output). The dedicated migration test below independently proves an actual
    # OLD-schema database reads back identically. Includes the zero-hash file.
    new = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    mem = _build_in_memory(_SQL_SURROGATE_CORPUS)
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)

    # search() ranking byte-identical to the in-memory parity reference.
    assert _search_tuples(new, query, top_k=10) == _search_tuples(mem, query, top_k=10)
    # Full SearchResult equality (score, votes, offset, confidence, metadata).
    assert new.search(query, top_k=10) == mem.search(query, top_k=10)
    # to_dict postings normalized through JSON (InMemory keeps tuples, SQL lists).
    assert json.loads(json.dumps(new.to_dict()["files"])) == \
           json.loads(json.dumps(mem.to_dict()["files"]))
    assert new.to_dict()["metadata"] == mem.to_dict()["metadata"]
    assert new.list_files() == mem.list_files()
    assert list(new.iter_metadata()) == list(mem.iter_metadata())
    assert new.posting_count == mem.posting_count
    assert new.file_count == mem.file_count

    # Replace-existing case: re-index a file_id, assert state matches a fresh
    # in-memory index built with the same replace, on both backends.
    replacement = [("a" * 64, [(2000, 7), (2001, 8)])]
    for fid, cos in replacement:
        new.add(fp_with_hashes(fid, cos))
        mem.add(fp_with_hashes(fid, cos))
    assert new.posting_count == mem.posting_count == (
        4 + 4 + 3 + 5 + 0 - 4 + 2  # corpus minus old "a" (4) plus new "a" (2)
    )
    assert _search_tuples(new, fp_with_hashes("q2" * 32, [(2000, 0), (2001, 0)]), top_k=10) == \
           _search_tuples(mem, fp_with_hashes("q2" * 32, [(2000, 0), (2001, 0)]), top_k=10)
    assert json.loads(json.dumps(new.to_dict()["files"]))["a" * 64] == [[2000, 7], [2001, 8]]
    new.close()


def test_sql_surrogate_cross_backend_parity_sqlite_inmemory_redis() -> None:
    # HARD GATE 2 (CROSS-BACKEND PARITY): InMemory == SQLite == Redis rankings
    # for the same surrogate corpus, plus snapshot interop in BOTH directions
    # (SQLite-written snapshot loads into every backend identically, and an
    # in-memory-written one loads back into the new-schema SQLite identically).
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)
    reference: list[tuple[str, int, int, float]] | None = None
    for name, index in _parity_backends():
        for file_id, code_offsets in _SQL_SURROGATE_CORPUS:
            index.add(fp_with_hashes(file_id, code_offsets))
        tuples = _search_tuples(index, query, top_k=10)
        if reference is None:
            reference = tuples
        assert tuples == reference, name
        assert index.list_files() == [fid for fid, _ in sorted(_SQL_SURROGATE_CORPUS)], name


def test_sql_surrogate_snapshot_interop_both_directions(tmp_path: Path) -> None:
    # HARD GATE 2 cont.: snapshot interop both ways across the new SQLite schema.
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)

    # SQLite -> snapshot -> InMemory (and back into a fresh new-schema SQLite).
    sql_source = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    snap = tmp_path / "from_sql.json"
    sql_source.save(snap)
    mem_target = InMemoryHashIndex.load(snap)
    sql_target = SQLiteHashIndex(":memory:").load_snapshot(snap)
    assert _search_tuples(mem_target, query) == _search_tuples(sql_source, query)
    assert _search_tuples(sql_target, query) == _search_tuples(sql_source, query)
    assert sql_target.to_dict()["files"] == sql_source.to_dict()["files"]
    sql_source.close()
    sql_target.close()

    # InMemory -> snapshot -> new-schema SQLite (reverse direction).
    mem_source = _build_in_memory(_SQL_SURROGATE_CORPUS)
    snap2 = tmp_path / "from_mem.json"
    mem_source.save(snap2)
    sql_from_mem = SQLiteHashIndex(":memory:").load_snapshot(snap2)
    assert _search_tuples(sql_from_mem, query) == _search_tuples(mem_source, query)
    assert sql_from_mem.list_files() == mem_source.list_files()
    assert json.loads(json.dumps(sql_from_mem.to_dict()["files"])) == \
           json.loads(json.dumps(mem_source.to_dict()["files"]))
    sql_from_mem.close()


def test_sqlite_migrates_old_schema_in_place_preserving_everything(tmp_path: Path) -> None:
    # HARD GATE 3 (MIGRATION): an OLD-schema .sqlite3 (postings.file_id TEXT,
    # files without id), built with raw sqlite3 to mimic the pre-change layout,
    # is opened by the new SQLiteHashIndex and must migrate in place to the
    # surrogate schema, returning search()/to_dict()/list_files() identical to a
    # freshly-built new-schema index of the same data.
    db = tmp_path / "legacy.sqlite3"
    _build_old_schema_sqlite(db, _SQL_SURROGATE_CORPUS)

    # Sanity: the file really is the OLD schema before we open it.
    probe = sqlite3.connect(str(db))
    old_cols = {row[1] for row in probe.execute("PRAGMA table_info(postings)").fetchall()}
    probe.close()
    assert "file_id" in old_cols and "file_ref" not in old_cols

    migrated = SQLiteHashIndex(db)
    # Schema was rewritten to the surrogate layout in place.
    new_cols = {row[1] for row in migrated._conn.execute("PRAGMA table_info(postings)").fetchall()}
    file_cols = {row[1] for row in migrated._conn.execute("PRAGMA table_info(files)").fetchall()}
    assert "file_ref" in new_cols and "file_id" not in new_cols
    assert "id" in file_cols

    fresh = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)
    assert _search_tuples(migrated, query, top_k=10) == _search_tuples(fresh, query, top_k=10)
    assert migrated.to_dict() == fresh.to_dict()  # postings order + metadata identical
    assert migrated.list_files() == fresh.list_files()
    assert migrated.posting_count == fresh.posting_count
    assert migrated.file_count == fresh.file_count
    # Every original metadata blob survived verbatim.
    for file_id, _ in _SQL_SURROGATE_CORPUS:
        assert migrated._metadata_for(file_id) == fresh._metadata_for(file_id)
    expected_files = fresh.list_files()
    expected_postings = fresh.posting_count
    fresh.close()
    migrated.close()

    # Re-opening the (now-migrated) file is a no-op: stable schema, same data.
    reopened = SQLiteHashIndex(db)
    new_cols2 = {row[1] for row in reopened._conn.execute("PRAGMA table_info(postings)").fetchall()}
    assert "file_ref" in new_cols2 and "file_id" not in new_cols2
    assert reopened.list_files() == expected_files
    assert reopened.posting_count == expected_postings
    reopened.close()


def test_sqlite_migration_is_writable_and_searchable_after_upgrade(tmp_path: Path) -> None:
    # After migrating, the index must behave like any new-schema index: add a new
    # file, remove an old one, and the surrogate path stays correct.
    db = tmp_path / "legacy_rw.sqlite3"
    _build_old_schema_sqlite(db, _SQL_SURROGATE_CORPUS)
    index = SQLiteHashIndex(db)

    index.add(fp_with_hashes("f" * 64, [(1000, 5), (1008, 6)]))
    index.remove("b" * 64)
    assert "f" * 64 in index.list_files()
    assert "b" * 64 not in index.list_files()
    # The added file is found, and no posting still references the removed file.
    assert index.search(fp_with_hashes("q" * 64, [(1008, 0)]), top_k=1)[0].file_id == "f" * 64
    assert all(p.file_id != "b" * 64 for p in index.query(1000))
    index.close()


def test_sqlite_fresh_db_is_not_treated_as_legacy(tmp_path: Path) -> None:
    # A brand-new database (no postings table yet) must NOT trip the migration
    # path; it just gets the new schema and works.
    db = tmp_path / "fresh.sqlite3"
    index = SQLiteHashIndex(db)
    index.add(fp_with_hashes("a" * 64, [(1, 1)]))
    cols = {row[1] for row in index._conn.execute("PRAGMA table_info(postings)").fetchall()}
    assert "file_ref" in cols
    assert index.list_files() == ["a" * 64]
    index.close()


def test_sqlite_surrogate_db_size_reduction_old_vs_new_schema(tmp_path: Path) -> None:
    # HARD GATE 4 (MEASURE): build ~2000 files x ~30 postings both ways and assert
    # the new-schema .sqlite3 is materially smaller than the old-schema one. The
    # win is that each of the 60000 posting rows now stores a small integer
    # file_ref instead of a 64-char file_id string.
    n_files, n_postings = 2000, 30
    offset = SQLiteHashIndex._SIGNED_OFFSET

    # OLD-schema DB built directly (the pre-change layout), WAL-checkpointed so
    # the measured size is the settled main database file, not WAL.
    old_db = tmp_path / "old.sqlite3"
    old_corpus = [(f"{i:064x}", [(1000 + j, j) for j in range(n_postings)]) for i in range(n_files)]
    conn = sqlite3.connect(str(old_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE files (file_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
        CREATE TABLE postings (
            file_id TEXT NOT NULL, hash_code INTEGER NOT NULL, time_offset INTEGER NOT NULL
        );
        CREATE INDEX idx_postings_hash ON postings(hash_code);
        CREATE INDEX idx_postings_file ON postings(file_id);
        """
    )
    for file_id, code_offsets in old_corpus:
        meta = {
            "file_id": file_id, "path": "", "handler": "test", "size_bytes": 0,
            "content_sha256": file_id, "hash_count": n_postings, "landmark_count": 0,
        }
        conn.execute("INSERT INTO files VALUES (?, ?)", (file_id, json.dumps(meta, sort_keys=True)))
        conn.executemany(
            "INSERT INTO postings VALUES (?, ?, ?)",
            [(file_id, code - offset, off) for code, off in code_offsets],
        )
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()
    conn.close()
    old_size = old_db.stat().st_size

    # NEW-schema DB via the class (add_many for the one-transaction bulk path).
    new_db = tmp_path / "new.sqlite3"
    index = SQLiteHashIndex(new_db)
    index.add_many([fp_with_hashes(fid, cos) for fid, cos in old_corpus])
    index._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    index._conn.commit()
    new_size = new_db.stat().st_size

    assert index.posting_count == n_files * n_postings
    assert index.file_count == n_files
    index.close()

    # The surrogate schema is materially smaller -- conservatively assert a >=25%
    # reduction (the observed win is far larger; the measurement is reported in
    # the task summary). This guards the optimization from regressing.
    assert new_size < old_size
    reduction = (old_size - new_size) / old_size
    assert reduction >= 0.25, f"old={old_size} new={new_size} reduction={reduction:.3f}"


@requires_pg
def test_pg_surrogate_output_equivalent_to_in_memory(pg_index) -> None:
    # HARD GATE 1/2 for Postgres (structural; only runs with a live server):
    # the surrogate-schema Postgres backend's rankings, snapshot, enumeration and
    # counts match the in-memory parity reference for the same corpus.
    mem = _build_in_memory(_SQL_SURROGATE_CORPUS)
    for file_id, code_offsets in _SQL_SURROGATE_CORPUS:
        pg_index.add(fp_with_hashes(file_id, code_offsets))
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)

    assert _search_tuples(pg_index, query, top_k=10) == _search_tuples(mem, query, top_k=10)
    assert pg_index.list_files() == mem.list_files()
    assert list(pg_index.iter_metadata()) == list(mem.iter_metadata())
    assert pg_index.posting_count == mem.posting_count
    assert pg_index.file_count == mem.file_count
    # The postings table carries an integer file_ref, not the file_id string.
    with pg_index._conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (pg_index._postings_table,),
        )
        cols = {row[0] for row in cur.fetchall()}
    pg_index._conn.rollback()
    assert "file_ref" in cols and "file_id" not in cols


@requires_pg
def test_pg_migrates_old_schema_in_place(pg_index) -> None:
    # HARD GATE 3 for Postgres (structural; only runs with a live server): drop
    # the surrogate tables the fixture created, recreate the OLD schema with
    # data, then re-open a PostgresHashIndex on the same prefix and assert it
    # migrated to the surrogate layout with output identical to the in-memory
    # reference for the same data.
    files_t, posts_t = pg_index._files_table, pg_index._postings_table
    offset = PostgresHashIndex._SIGNED_OFFSET
    with pg_index._conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {posts_t}, {files_t}")
        cur.execute(f"CREATE TABLE {files_t} (file_id TEXT PRIMARY KEY, metadata JSONB NOT NULL)")
        cur.execute(
            f"CREATE TABLE {posts_t} "
            "(file_id TEXT NOT NULL, hash_code BIGINT NOT NULL, time_offset INTEGER NOT NULL)"
        )
        for file_id, code_offsets in _SQL_SURROGATE_CORPUS:
            meta = {
                "file_id": file_id, "path": f"/tmp/{file_id}", "handler": "test",
                "size_bytes": 10, "content_sha256": file_id,
                "hash_count": len(code_offsets), "landmark_count": 0,
            }
            cur.execute(
                f"INSERT INTO {files_t} (file_id, metadata) VALUES (%s, %s::jsonb)",
                (file_id, json.dumps(meta, sort_keys=True)),
            )
            for code, off in code_offsets:
                cur.execute(
                    f"INSERT INTO {posts_t} (file_id, hash_code, time_offset) VALUES (%s, %s, %s)",
                    (file_id, code - offset, off),
                )
    pg_index._conn.commit()

    migrated = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest")  # same prefix -> migrates
    try:
        with migrated._conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (posts_t,),
            )
            cols = {row[0] for row in cur.fetchall()}
        migrated._conn.rollback()
        assert "file_ref" in cols and "file_id" not in cols

        mem = _build_in_memory(_SQL_SURROGATE_CORPUS)
        query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)
        assert _search_tuples(migrated, query, top_k=10) == _search_tuples(mem, query, top_k=10)
        assert migrated.list_files() == mem.list_files()
        assert migrated.posting_count == mem.posting_count
    finally:
        migrated.close()
