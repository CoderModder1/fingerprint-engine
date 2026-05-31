"""SQLite-backed hash index (stdlib, file-persistent)."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from ..models import (
    Fingerprint,
    IndexPosting,
)
from ._common import (
    SNAPSHOT_FORMAT_VERSION_KEY,
    _index_metadata,
    _synchronized,
)
from .base import HashIndex

logger = logging.getLogger(__name__)


class SQLiteHashIndex(HashIndex):
    """SQLite-backed hash index: zero-dependency (stdlib), file-persistent.

    Implements the same :class:`HashIndex` contract and inherits the shared
    ``search``/``save``/``load_snapshot``. Pass ``":memory:"`` for an ephemeral
    in-process database (used by tests) or a file path for persistence; or inject
    an existing ``sqlite3.Connection``.

    Storage layout (internal, output-preserving): the 64-char SHA-256 ``file_id``
    is the dominant per-posting cost when stored verbatim on every posting row.
    Instead a normalized ``files`` row maps each ``file_id`` to a small integer
    surrogate (its ``INTEGER PRIMARY KEY`` rowid), and each posting stores that
    integer ``file_ref`` foreign key rather than the 64-char string. The
    surrogate is mapped back to the original ``file_id`` string only at the
    query/aggregate/snapshot boundary (every read JOINs ``postings`` to
    ``files``), so :meth:`search`, :meth:`to_dict`, :meth:`list_files`,
    :meth:`iter_metadata`, :meth:`query`, :meth:`query_many`, ``posting_count``,
    ``file_count`` and cross-backend parity are all BYTE-IDENTICAL to storing the
    string verbatim. This mirrors the in-memory backend's surrogate concept.

    ``postings`` is indexed on ``hash_code`` (fast ``query``) and ``file_ref``
    (fast ``remove``/aggregation join).

    Migration: a database written by a PRIOR version of this class has the OLD
    schema (``postings.file_id TEXT``, a ``files`` table with no ``id`` column).
    :meth:`_init_schema` DETECTS that layout and migrates it in place, in a single
    transaction, exactly once on open -- preserving every ``file_id``, metadata
    blob, and posting -- so an existing persistent ``.sqlite3`` keeps working
    transparently. See :meth:`_migrate_legacy_schema`.

    Concurrency contract: file-backed databases run in WAL journal mode, which
    allows many concurrent readers alongside a single writer; a 5s
    ``busy_timeout`` lets a blocked writer wait instead of failing immediately
    with "database is locked". This is a single-writer model -- SQLite still
    serializes writes. WAL is a harmless no-op for a ``":memory:"`` database.
    """

    # Hash codes are unsigned 64-bit (0 .. 2**64-1) but SQLite INTEGER is signed
    # 64-bit (max 2**63-1), so we store them shifted into signed range with this
    # reversible offset. Keeps fast integer indexing without overflow.
    _SIGNED_OFFSET = 1 << 63

    def __init__(
        self,
        database: str | Path = "fingerprint_index.sqlite3",
        connection: sqlite3.Connection | None = None,
    ) -> None:
        # Serializes access to the single shared connection (F3). The connection
        # is opened check_same_thread=False so the FastAPI threadpool can touch it
        # from many threads; a SQLite connection is one serial command stream, so
        # the multi-statement critical sections (the _aggregate temp table, the
        # write transactions) MUST be serialized or concurrent requests interleave
        # and corrupt results. RLock (not Lock) because add() calls remove() and
        # _record_format_version(), and search() composes query_many()+_aggregate().
        # Uncontended in the single-threaded path, so output is byte-identical.
        self._lock = threading.RLock()
        if connection is not None:
            self._conn = connection
        else:
            self._conn = sqlite3.connect(str(database), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        # WAL enables concurrent readers + one writer and is a no-op for
        # ":memory:"; busy_timeout avoids immediate "database is locked" errors
        # when a writer briefly contends with another connection.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        # synchronous=NORMAL is the recommended durable pairing with WAL: the
        # WAL still guarantees crash consistency, only losing the very last
        # committed transaction on an OS/power crash (never corruption). This
        # cuts the fsync-per-commit cost that bottlenecks bulk add_many.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # An existing database written by the OLD schema (postings.file_id TEXT,
        # files without an id surrogate) is migrated in place, once, before the
        # new tables/indexes are (idempotently) ensured below.
        self._migrate_legacy_schema_if_present()
        # files.id is an INTEGER PRIMARY KEY, i.e. an alias for the rowid -- the
        # small integer surrogate stored in postings.file_ref. file_id stays
        # UNIQUE so upsert-by-file_id resolves a stable id; metadata is the same
        # JSON blob as before.
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id       INTEGER PRIMARY KEY,
                file_id  TEXT UNIQUE NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS postings (
                file_ref    INTEGER NOT NULL,
                hash_code   INTEGER NOT NULL,
                time_offset INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_postings_hash ON postings(hash_code);
            CREATE INDEX IF NOT EXISTS idx_postings_file ON postings(file_ref);
            -- Side table recording the corpus hash-format version (F2). A row is
            -- written only for a NON-default corpus (see _persist_format_version);
            -- absent => default version, so a default store has an empty meta and
            -- is byte-identical to before. Never read by to_dict/search.
            CREATE TABLE IF NOT EXISTS index_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        # Restore the durably-stored hash format version (F2) so a reopened store
        # reports the corpus version rather than silently defaulting to baseline.
        # Absent (a fresh DB, or a legacy DB written before this table) -> stay at
        # the unpinned default, matching prior behaviour until the next add.
        row = self._conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (SNAPSHOT_FORMAT_VERSION_KEY,)
        ).fetchone()
        if row is not None:
            self._load_persisted_format_version(row[0])

    def _persist_format_version(self, version: int) -> None:
        # F2: stamp the pinned (non-default) version into the side meta table so a
        # reopen restores it. Self-contained commit (only ever runs once, on first
        # pin of a non-default corpus) so it survives even if the triggering add
        # later fails, and never leaves a dangling transaction.
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                (SNAPSHOT_FORMAT_VERSION_KEY, str(version)),
            )
            self._conn.commit()

    def _postings_columns(self) -> set[str]:
        """Column names of the existing ``postings`` table (empty set if absent)."""

        return {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(postings)").fetchall()
        }

    def _migrate_legacy_schema_if_present(self) -> None:
        """One-time, transactional, in-place upgrade of an OLD-schema database.

        The pre-surrogate schema stored ``postings.file_id TEXT`` and a ``files``
        table with no ``id`` column. We detect that exact shape -- a ``postings``
        table that HAS a ``file_id`` column and LACKS ``file_ref`` -- and rewrite
        it to the surrogate layout: build the new ``files`` (with an INTEGER
        PRIMARY KEY id) and ``postings`` (with file_ref), populate the surrogate
        ids from the distinct file_ids, copy every posting across resolving its
        file_id to the new id, then drop the old tables and rename the new ones
        into place. Every file_id, metadata blob, and posting (and its insertion
        order, preserved by copying ORDER BY the old rowid) survives unchanged, so
        a migrated index is output-identical to one freshly built from the same
        data.

        A brand-new or already-migrated database (no ``postings`` table, or one
        that already has ``file_ref``) is left untouched -- this is a no-op except
        the one-time legacy upgrade. The whole rewrite runs in a single
        transaction: a failure rolls back to the original old-schema database
        rather than leaving a half-migrated one.
        """

        columns = self._postings_columns()
        if not columns or "file_ref" in columns or "file_id" not in columns:
            # No postings table yet (fresh DB), or already the new schema, or an
            # unrecognized shape we must not touch -- nothing to migrate.
            return
        # Stamp a marker so the intent is greppable in the file; harmless if the
        # rewrite below is interrupted (the next open re-detects the old schema).
        self._conn.execute("PRAGMA legacy_alter_table=OFF")
        try:
            self._conn.execute("BEGIN")
            self._conn.executescript(
                """
                CREATE TABLE files_new (
                    id       INTEGER PRIMARY KEY,
                    file_id  TEXT UNIQUE NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE TABLE postings_new (
                    file_ref    INTEGER NOT NULL,
                    hash_code   INTEGER NOT NULL,
                    time_offset INTEGER NOT NULL
                );
                -- Surrogate ids are assigned by INTEGER PRIMARY KEY rowid as the
                -- distinct file_ids are inserted (ordered by the old files rowid
                -- for a deterministic assignment).
                INSERT INTO files_new (file_id, metadata)
                    SELECT file_id, metadata FROM files ORDER BY rowid;
                -- Copy every posting, resolving its file_id to the new surrogate.
                -- ORDER BY the old posting rowid preserves per-file insertion
                -- order, so to_dict()'s ORDER BY rowid stays byte-identical.
                INSERT INTO postings_new (file_ref, hash_code, time_offset)
                    SELECT f.id, p.hash_code, p.time_offset
                    FROM postings p JOIN files_new f ON f.file_id = p.file_id
                    ORDER BY p.rowid;
                DROP TABLE postings;
                DROP TABLE files;
                ALTER TABLE files_new RENAME TO files;
                ALTER TABLE postings_new RENAME TO postings;
                """
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    @classmethod
    def _encode(cls, hash_code: int) -> int:
        return int(hash_code) - cls._SIGNED_OFFSET

    @classmethod
    def _decode(cls, stored: int) -> int:
        return int(stored) + cls._SIGNED_OFFSET

    @property
    @_synchronized
    def file_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    @property
    @_synchronized
    def posting_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0])

    @_synchronized
    def add(self, fingerprint: Fingerprint) -> None:
        self._record_format_version(fingerprint)
        self.remove(fingerprint.file_id)
        metadata = _index_metadata(fingerprint)
        try:
            # remove() above deleted any prior files row, so this INSERT allocates
            # a fresh surrogate id (cursor.lastrowid). Postings carry that
            # file_ref, not the 64-char file_id string.
            cursor = self._conn.execute(
                "INSERT INTO files (file_id, metadata) VALUES (?, ?)",
                (fingerprint.file_id, json.dumps(metadata, sort_keys=True)),
            )
            file_ref = cursor.lastrowid
            self._conn.executemany(
                "INSERT INTO postings (file_ref, hash_code, time_offset) VALUES (?, ?, ?)",
                [
                    (file_ref, self._encode(item.hash_code), int(item.time_offset))
                    for item in fingerprint.hashes
                ],
            )
            self._conn.commit()
        except BaseException:
            # Without this, a failure after the files INSERT (e.g. a posting
            # insert error) left the files row in an open, uncommitted
            # transaction that the NEXT successful commit silently flushed --
            # a committed phantom file with zero postings. Mirror add_many():
            # roll the whole add back so it is all-or-nothing.
            self._conn.rollback()
            raise

    @_synchronized
    def add_many(self, fingerprints: Iterable[Fingerprint]) -> None:
        """Ingest the whole batch in ONE transaction (one commit, not per file).

        Equivalent to per-item :meth:`add`: each ``file_id`` is removed before it
        is (re)inserted, so a duplicate ``file_id`` within the batch keeps only
        the LAST fingerprint and any pre-batch rows for it are deleted exactly
        once -- identical resulting postings, metadata, and search results. The
        single commit (instead of one fsync-bound commit per file) plus a single
        ``executemany`` for all postings is the bulk-ingest win.
        """

        # Fail-closed PRE-CHECK (no pin/persist) so a cross-format batch raises
        # before any write OR version stamp -- true all-or-nothing (F1). Without
        # it the first member would pin+persist the version off a batch that then
        # rolls back, leaving an empty store reporting a non-default version.
        fingerprints = list(fingerprints)
        self._validate_batch_format(fingerprints)
        # Last-wins per file_id: a sequential add() of a repeated file_id removes
        # the earlier copy, so only the final fingerprint survives. Collapsing
        # here also avoids the files PRIMARY KEY conflict a naive re-insert hits.
        survivors: dict[str, Fingerprint] = {}
        for fingerprint in fingerprints:
            self._record_format_version(fingerprint)
            survivors[fingerprint.file_id] = fingerprint
        if not survivors:
            return

        # Per file: its metadata blob plus its encoded postings. The surrogate
        # file_ref is filled in below, once the files row is inserted and its id
        # (lastrowid) is known -- postings can only carry a ref after the id
        # exists, so they are staged per file and assembled into one bulk insert.
        staged: list[tuple[str, str, list[tuple[int, int]]]] = []
        for file_id, fingerprint in survivors.items():
            metadata = _index_metadata(fingerprint)
            staged.append((
                file_id,
                json.dumps(metadata, sort_keys=True),
                [(self._encode(item.hash_code), int(item.time_offset)) for item in fingerprint.hashes],
            ))

        try:
            # remove() effect for every target file_id (pre-batch rows). Postings
            # are keyed by the surrogate, so resolve each id and delete by ref
            # (one batched lookup of the existing ids, then a batched delete).
            file_ids = list(survivors)
            placeholders = ",".join("?" * len(file_ids))
            old_refs = [
                row[0]
                for row in self._conn.execute(
                    f"SELECT id FROM files WHERE file_id IN ({placeholders})", file_ids
                ).fetchall()
            ]
            if old_refs:
                self._conn.executemany(
                    "DELETE FROM postings WHERE file_ref = ?", [(ref,) for ref in old_refs]
                )
            self._conn.executemany(
                "DELETE FROM files WHERE file_id = ?", [(fid,) for fid in file_ids]
            )
            # Insert each files row (capturing its fresh surrogate id) then stage
            # that file's postings with the resolved file_ref. One executemany
            # then bulk-inserts every posting in a single statement.
            posting_rows: list[tuple[int, int, int]] = []
            for file_id, meta_json, encoded in staged:
                cursor = self._conn.execute(
                    "INSERT INTO files (file_id, metadata) VALUES (?, ?)", (file_id, meta_json)
                )
                file_ref = cursor.lastrowid
                assert file_ref is not None  # an INTEGER PRIMARY KEY insert always sets it
                posting_rows.extend((file_ref, code, offset) for code, offset in encoded)
            self._conn.executemany(
                "INSERT INTO postings (file_ref, hash_code, time_offset) VALUES (?, ?, ?)",
                posting_rows,
            )
        except BaseException:
            # Keep the all-or-nothing contract: a mid-batch failure rolls the
            # whole transaction back instead of leaving a partial ingest.
            self._conn.rollback()
            raise
        self._conn.commit()

    @_synchronized
    def remove(self, file_id: str) -> None:
        # Resolve the surrogate first so postings (keyed by file_ref) can be
        # deleted, then drop the files row. A subselect keeps it one round-trip.
        self._conn.execute(
            "DELETE FROM postings WHERE file_ref = (SELECT id FROM files WHERE file_id = ?)",
            (file_id,),
        )
        self._conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        self._conn.commit()

    @_synchronized
    def query(self, hash_code: int) -> list[IndexPosting]:
        code = int(hash_code)
        # JOIN postings -> files to recover the original str file_id from the
        # surrogate file_ref; the returned IndexPosting values are unchanged.
        rows = self._conn.execute(
            "SELECT f.file_id, p.time_offset FROM postings p "
            "JOIN files f ON f.id = p.file_ref WHERE p.hash_code = ?",
            (self._encode(code),),
        ).fetchall()
        return [
            IndexPosting(file_id=row[0], hash_code=code, time_offset=int(row[1]))
            for row in rows
        ]

    @_synchronized
    def query_many(self, hash_codes: Iterable[int]) -> dict[int, list[IndexPosting]]:
        codes = list({int(c) for c in hash_codes})
        results: dict[int, list[IndexPosting]] = {code: [] for code in codes}
        chunk = 500  # stay well under SQLITE_MAX_VARIABLE_NUMBER
        for start in range(0, len(codes), chunk):
            batch = [self._encode(code) for code in codes[start:start + chunk]]
            placeholders = ",".join("?" * len(batch))
            # JOIN to files to map each posting's surrogate file_ref back to the
            # original str file_id, preserving the IndexPosting return contract.
            rows = self._conn.execute(
                f"SELECT p.hash_code, f.file_id, p.time_offset FROM postings p "
                f"JOIN files f ON f.id = p.file_ref "
                f"WHERE p.hash_code IN ({placeholders})",
                batch,
            ).fetchall()
            for stored, file_id, time_offset in rows:
                code = self._decode(stored)
                results[code].append(
                    IndexPosting(file_id=file_id, hash_code=code, time_offset=int(time_offset))
                )
        return results

    @_synchronized
    def _aggregate(
        self,
        fingerprint: Fingerprint,
        offset_tolerance: int = 0,
        candidates: set[str] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Aggregate the offset histogram server-side via a single SQL pass.

        Re-entrancy/concurrency (F3): the shared ``_query`` temp table below is a
        connection-global scratch space, so this whole method is serialized by the
        per-index ``RLock`` (via ``@_synchronized``) -- without it, two threads in
        the FastAPI threadpool would interleave the DELETE/INSERT/SELECT steps on
        the one connection and cross-contaminate or corrupt rankings. The lock is
        uncontended single-threaded, so the produced rows are byte-identical.

        Loads the query's (hash_code, offset) pairs into a temp table, joins to
        the postings, and groups by (file_ref, delta) -- so only per-file
        aggregates cross the boundary, not millions of postings. Grouping on the
        small integer ``file_ref`` (the surrogate) is cheaper than grouping on the
        64-char file_id; the surrogate is resolved back to the original str
        ``file_id`` by a single JOIN to ``files`` in the final SELECT, so the
        result is keyed by the SAME ``file_id`` string as every other backend.

        With ``offset_tolerance == 0`` (the default) the winning bin is picked
        server-side as votes DESC then delta ASC, matching the base in-memory
        tie-break exactly -- this path is unchanged and BYTE-IDENTICAL to before
        the option existed (partitioning by ``file_ref`` instead of the file_id
        string is a 1:1 relabelling of the same partitions, so the per-file
        winning row is unchanged). With ``> 0`` the SQL still groups to
        per-(file, delta) bins server-side (only the compact histogram crosses the
        boundary, never raw postings), but the banded winner is chosen in Python
        via the shared :meth:`_banded_winner`, so every backend bands identically.

        ``candidates`` is the OPT-IN prefilter set: ``None`` (the default) returns
        the aggregate for every matched file, byte-identical to before the
        prefilter existed; a set keeps only those ``file_id``. The restriction is
        applied to the per-file aggregate rows (their grouping is per-file, so
        dropping a non-candidate never alters a retained file's row), so a
        candidate superset of the true top-k yields the identical ranking.
        """

        pairs = [(self._encode(h.hash_code), int(h.time_offset)) for h in fingerprint.hashes]
        if not pairs:
            return {}
        banded = offset_tolerance > 0
        try:
            self._conn.execute("CREATE TEMP TABLE IF NOT EXISTS _query (hash_code INTEGER, qoff INTEGER)")
            self._conn.execute("DELETE FROM _query")
            self._conn.executemany("INSERT INTO _query (hash_code, qoff) VALUES (?, ?)", pairs)
            if banded:
                rows = self._conn.execute(
                    """
                    WITH matches AS (
                        SELECT p.file_ref AS file_ref, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                        FROM postings p JOIN _query q ON p.hash_code = q.hash_code
                    ),
                    bins AS (SELECT file_ref, delta, COUNT(*) AS votes FROM matches GROUP BY file_ref, delta),
                    totals AS (
                        SELECT file_ref, COUNT(*) AS total_votes, COUNT(DISTINCT hc) AS uniq
                        FROM matches GROUP BY file_ref
                    )
                    SELECT f.file_id, b.delta, b.votes, t.total_votes, t.uniq
                    FROM bins b JOIN totals t ON b.file_ref = t.file_ref
                    JOIN files f ON f.id = b.file_ref
                    """
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    WITH matches AS (
                        SELECT p.file_ref AS file_ref, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                        FROM postings p JOIN _query q ON p.hash_code = q.hash_code
                    ),
                    bins AS (SELECT file_ref, delta, COUNT(*) AS votes FROM matches GROUP BY file_ref, delta),
                    ranked AS (
                        SELECT file_ref, delta, votes,
                               ROW_NUMBER() OVER (PARTITION BY file_ref ORDER BY votes DESC, delta ASC) AS rn
                        FROM bins
                    ),
                    totals AS (
                        SELECT file_ref, COUNT(*) AS total_votes, COUNT(DISTINCT hc) AS uniq
                        FROM matches GROUP BY file_ref
                    )
                    SELECT f.file_id, r.delta, r.votes, t.total_votes, t.uniq
                    FROM ranked r JOIN totals t ON r.file_ref = t.file_ref
                    JOIN files f ON f.id = r.file_ref
                    WHERE r.rn = 1
                    """
                ).fetchall()
            self._conn.execute("DELETE FROM _query")
        finally:
            # The DML above (CREATE TEMP/DELETE/INSERT) opens an implicit write
            # transaction; commit so this read-only path never leaves the
            # connection holding a write lock for the rest of its lifetime.
            self._conn.commit()
        if candidates is not None:
            rows = [row for row in rows if str(row[0]) in candidates]
        if banded:
            return self._reduce_banded_rows(rows, offset_tolerance)
        return {
            file_id: (int(delta), int(votes), int(total), int(uniq))
            for file_id, delta, votes, total, uniq in rows
        }

    @_synchronized
    def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
        file_total = self.file_count
        if file_total == 0:
            return 0
        threshold = max_df_ratio * file_total
        # Document frequency is distinct surrogates touching a code (1:1 with
        # distinct file_ids), so COUNT(DISTINCT file_ref) is the same df test.
        cursor = self._conn.execute(
            "DELETE FROM postings WHERE hash_code IN ("
            "  SELECT hash_code FROM postings GROUP BY hash_code "
            "  HAVING COUNT(DISTINCT file_ref) > ?)",
            (threshold,),
        )
        removed = cursor.rowcount
        if removed:
            # Remaining postings per file, resolving the surrogate back to file_id.
            counts = dict(
                self._conn.execute(
                    "SELECT f.file_id, COUNT(*) FROM postings p "
                    "JOIN files f ON f.id = p.file_ref GROUP BY p.file_ref"
                ).fetchall()
            )
            for file_id, meta_json in self._conn.execute("SELECT file_id, metadata FROM files").fetchall():
                metadata = json.loads(meta_json)
                new_count = int(counts.get(file_id, 0))
                if metadata.get("hash_count") != new_count:
                    metadata["hash_count"] = new_count
                    self._conn.execute(
                        "UPDATE files SET metadata = ? WHERE file_id = ?",
                        (json.dumps(metadata, sort_keys=True), file_id),
                    )
        self._conn.commit()
        return removed

    @_synchronized
    def _metadata_for(self, file_id: str) -> dict:
        row = self._conn.execute(
            "SELECT metadata FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        if not row:
            return {}
        try:
            data = json.loads(row[0])
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    @_synchronized
    def list_files(self) -> list[str]:
        # ORDER BY in SQL gives the same ascending order as the base sorted()
        # contract without a Python-side sort of the whole id set.
        return [
            row[0]
            for row in self._conn.execute("SELECT file_id FROM files ORDER BY file_id").fetchall()
        ]

    @_synchronized
    def contains(self, file_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM files WHERE file_id = ? LIMIT 1", (file_id,)
        ).fetchone()
        return row is not None

    @_synchronized
    def to_dict(self) -> dict[str, object]:
        files: dict[str, list[list[int]]] = {}
        metadata: dict[str, dict] = {}
        # Carry the surrogate id so postings can be read by file_ref; ORDER BY
        # rowid is preserved so per-file posting order is byte-identical to before.
        for file_ref, file_id in self._conn.execute("SELECT id, file_id FROM files").fetchall():
            entries = [
                [self._decode(hash_code), int(time_offset)]
                for hash_code, time_offset in self._conn.execute(
                    "SELECT hash_code, time_offset FROM postings "
                    "WHERE file_ref = ? ORDER BY rowid",
                    (file_ref,),
                ).fetchall()
            ]
            files[file_id] = entries
            metadata[file_id] = self._metadata_for(file_id)
        return {"backend": "sqlite", "files": files, "metadata": metadata}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteHashIndex:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
