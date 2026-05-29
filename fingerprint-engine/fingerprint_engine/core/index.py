"""Hash index interfaces and the default in-memory backend."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .models import Calibration, ConstellationHash, Fingerprint, IndexPosting, SearchResult


class HashIndex(ABC):
    """Storage-agnostic contract for searchable fingerprint postings."""

    @property
    @abstractmethod
    def file_count(self) -> int:
        """Number of indexed files."""

    @property
    @abstractmethod
    def posting_count(self) -> int:
        """Total number of postings across all indexed files."""

    @abstractmethod
    def add(self, fingerprint: Fingerprint) -> None:
        """Add or replace a fingerprint in the index."""

    @abstractmethod
    def remove(self, file_id: str) -> None:
        """Remove all postings for a file."""

    @abstractmethod
    def query(self, hash_code: int) -> list[IndexPosting]:
        """Return postings for one hash code."""

    @abstractmethod
    def _metadata_for(self, file_id: str) -> dict:
        """Return stored metadata for a file (empty dict if unknown)."""

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Return a portable ``{backend, files, metadata}`` snapshot of the index."""

    def query_many(self, hash_codes: Iterable[int]) -> dict[int, list[IndexPosting]]:
        """Return postings for many hash codes in one batch: ``{code: postings}``.

        The default fans out to :meth:`query`. Storage backends SHOULD override
        this with a single batched round-trip -- the per-code default makes
        :meth:`search` issue one lookup per query hash (thousands per search),
        which is fine in-memory but catastrophic for SQL backends.
        """

        return {int(code): self.query(int(code)) for code in {int(c) for c in hash_codes}}

    def search(
        self,
        fingerprint: Fingerprint,
        top_k: int = 10,
        calibration: Calibration | None = None,
    ) -> list[SearchResult]:
        """Return ranked matches via Shazam-style offset-histogram alignment.

        Aggregation (per file: winning offset, aligned/total votes, unique
        hashes) is delegated to :meth:`_aggregate` so SQL backends can compute it
        server-side; scoring/calibration/ranking is shared here, so every backend
        produces identical results. Each result carries a handler-independent
        ``confidence`` in [0, 1] (aligned votes / the smaller fingerprint's hash
        count); when a :class:`Calibration` is supplied, results below its
        per-handler threshold are dropped.
        """

        return self._finalize(fingerprint, self._aggregate(fingerprint), top_k, calibration)

    def _aggregate(self, fingerprint: Fingerprint) -> dict[str, tuple[int, int, int, int]]:
        """Per-file ``(offset, aligned_votes, total_votes, unique_hashes)``.

        Default in-memory aggregation over a batched :meth:`query_many` fetch.
        The winning offset is the bin with the most votes, ties broken by
        smallest offset -- a deterministic rule SQL backends replicate exactly.
        """

        offset_histograms: dict[str, Counter[int]] = defaultdict(Counter)
        total_votes: Counter[str] = Counter()
        unique_hashes: dict[str, set[int]] = defaultdict(set)

        postings_by_code = self.query_many({qh.hash_code for qh in fingerprint.hashes})
        for query_hash in fingerprint.hashes:
            for posting in postings_by_code.get(query_hash.hash_code, ()):
                offset = posting.time_offset - query_hash.time_offset
                offset_histograms[posting.file_id][offset] += 1
                total_votes[posting.file_id] += 1
                unique_hashes[posting.file_id].add(query_hash.hash_code)

        aggregates: dict[str, tuple[int, int, int, int]] = {}
        for file_id, histogram in offset_histograms.items():
            if not histogram:
                continue
            offset, aligned = max(histogram.items(), key=lambda kv: (kv[1], -kv[0]))
            aggregates[file_id] = (offset, aligned, total_votes[file_id], len(unique_hashes[file_id]))
        return aggregates

    def _finalize(
        self,
        fingerprint: Fingerprint,
        aggregates: dict[str, tuple[int, int, int, int]],
        top_k: int,
        calibration: Calibration | None,
    ) -> list[SearchResult]:
        """Score, calibrate, and rank per-file aggregates (shared by all backends)."""

        results: list[SearchResult] = []
        query_hash_count = max(1, fingerprint.hash_count)
        for file_id, (offset, aligned_votes, total, unique) in aggregates.items():
            alignment_ratio = aligned_votes / max(1, total)
            coverage_ratio = unique / query_hash_count
            score = aligned_votes + (0.30 * unique) + (5.0 * alignment_ratio) + (2.0 * coverage_ratio)
            metadata = dict(self._metadata_for(file_id))
            # Handler-independent confidence: fraction of the smaller fingerprint's
            # hashes that aligned at the winning offset.
            target_hash_count = int(metadata.get("hash_count") or query_hash_count)
            confidence = min(1.0, aligned_votes / max(1, min(query_hash_count, target_hash_count)))
            if calibration is not None and not calibration.accepts(
                str(metadata.get("handler", fingerprint.handler)), confidence
            ):
                continue
            results.append(
                SearchResult(
                    file_id=file_id,
                    score=round(float(score), 6),
                    aligned_votes=int(aligned_votes),
                    total_votes=int(total),
                    unique_hashes=int(unique),
                    offset=int(offset),
                    confidence=round(float(confidence), 6),
                    metadata=metadata,
                )
            )

        results.sort(
            key=lambda item: (
                -item.score,
                -item.aligned_votes,
                -item.unique_hashes,
                item.file_id,
            )
        )
        return results[:top_k]

    def save(self, path: str | Path) -> None:
        """Write a portable JSON snapshot (same schema for every backend)."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, sort_keys=True, separators=(",", ":"))

    def load_snapshot(self, path: str | Path) -> HashIndex:
        """Bulk-load a JSON snapshot (from any backend's ``save``) via ``add``."""

        source = Path(path)
        if not source.exists():
            return self
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        files = data.get("files", {}) if isinstance(data, dict) else {}
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        if not isinstance(files, dict):
            raise ValueError("invalid index snapshot: files must be a mapping")
        for file_id, entries in files.items():
            if not isinstance(entries, list):
                continue
            hashes = [
                ConstellationHash(
                    hash_code=int(entry[0]),
                    time_offset=int(entry[1]),
                    anchor_time=int(entry[1]),
                    target_time=int(entry[1]),
                    freq1=0,
                    freq2=0,
                    delta_t=0,
                )
                for entry in entries
                if isinstance(entry, (list, tuple)) and len(entry) == 2
            ]
            meta = metadata.get(file_id, {}) if isinstance(metadata, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            self.add(
                Fingerprint(
                    file_id=str(file_id),
                    path=str(meta.get("path", "")),
                    handler=str(meta.get("handler", "")),
                    size_bytes=int(meta.get("size_bytes", 0) or 0),
                    content_sha256=str(meta.get("content_sha256", file_id)),
                    config={},
                    hashes=hashes,
                    metadata={
                        key: value
                        for key, value in meta.items()
                        if key not in {
                            "file_id", "path", "handler", "size_bytes",
                            "content_sha256", "hash_count", "landmark_count",
                        }
                    },
                )
            )
        return self

    def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
        """Remove postings for non-discriminative "stop" hash codes.

        A hash code present in more than ``max_df_ratio`` of indexed files
        carries little discriminative signal but dominates query cost and storage
        (its posting list is huge). Pruning these speeds up search and shrinks the
        index. Each affected file's stored ``hash_count`` is updated to its
        remaining postings so confidence stays calibrated (a self-match stays
        ~1.0). Returns the number of postings removed. Backends override this;
        the default declines.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support prune_stop_hashes; "
            "rebuild from a snapshot of a pruned index instead"
        )


class InMemoryHashIndex(HashIndex):
    """Dict-backed hash index with Shazam-style offset alignment scoring."""

    def __init__(self) -> None:
        self._postings: defaultdict[int, list[IndexPosting]] = defaultdict(list)
        self._file_entries: dict[str, list[tuple[int, int]]] = {}
        self._metadata: dict[str, dict[str, object]] = {}

    @property
    def file_count(self) -> int:
        return len(self._file_entries)

    @property
    def posting_count(self) -> int:
        return sum(len(postings) for postings in self._postings.values())

    def add(self, fingerprint: Fingerprint) -> None:
        self.remove(fingerprint.file_id)
        entries = [(item.hash_code, item.time_offset) for item in fingerprint.hashes]
        self._file_entries[fingerprint.file_id] = entries
        self._metadata[fingerprint.file_id] = {
            "file_id": fingerprint.file_id,
            "path": fingerprint.path,
            "handler": fingerprint.handler,
            "size_bytes": fingerprint.size_bytes,
            "content_sha256": fingerprint.content_sha256,
            "hash_count": fingerprint.hash_count,
            "landmark_count": fingerprint.landmark_count,
            **fingerprint.metadata,
        }
        for hash_code, time_offset in entries:
            self._postings[hash_code].append(
                IndexPosting(
                    file_id=fingerprint.file_id,
                    hash_code=hash_code,
                    time_offset=time_offset,
                )
            )

    def remove(self, file_id: str) -> None:
        if file_id not in self._file_entries:
            return

        for hash_code, _time_offset in self._file_entries[file_id]:
            self._postings[hash_code] = [
                posting
                for posting in self._postings[hash_code]
                if posting.file_id != file_id
            ]
            if not self._postings[hash_code]:
                del self._postings[hash_code]

        del self._file_entries[file_id]
        self._metadata.pop(file_id, None)

    def query(self, hash_code: int) -> list[IndexPosting]:
        return list(self._postings.get(int(hash_code), []))

    def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
        file_total = len(self._file_entries)
        if file_total == 0:
            return 0
        threshold = max_df_ratio * file_total
        stop = {
            code
            for code, postings in self._postings.items()
            if len({posting.file_id for posting in postings}) > threshold
        }
        if not stop:
            return 0
        removed = sum(len(self._postings.pop(code)) for code in stop)
        for file_id, entries in self._file_entries.items():
            kept = [(h, t) for (h, t) in entries if h not in stop]
            if len(kept) != len(entries):
                self._file_entries[file_id] = kept
                if file_id in self._metadata:
                    self._metadata[file_id]["hash_count"] = len(kept)
        return removed

    def _metadata_for(self, file_id: str) -> dict:
        return dict(self._metadata.get(file_id, {}))

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": "in_memory",
            "files": self._file_entries,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> InMemoryHashIndex:
        index = cls()
        files = data.get("files", {})
        if not isinstance(files, dict):
            raise ValueError("invalid index: files must be a mapping")
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            index._metadata = {
                str(file_id): dict(value) if isinstance(value, dict) else {}
                for file_id, value in metadata.items()
            }

        for file_id, entries in files.items():
            normalized_entries: list[tuple[int, int]] = []
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, list | tuple) or len(entry) != 2:
                    continue
                hash_code = int(entry[0])
                time_offset = int(entry[1])
                normalized_entries.append((hash_code, time_offset))
                index._postings[hash_code].append(
                    IndexPosting(
                        file_id=str(file_id),
                        hash_code=hash_code,
                        time_offset=time_offset,
                    )
                )
            index._file_entries[str(file_id)] = normalized_entries
        return index

    @classmethod
    def load(cls, path: str | Path) -> InMemoryHashIndex:
        source = Path(path)
        if not source.exists():
            return cls()
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("invalid index file")
        return cls.from_dict(data)


class RedisHashIndex(HashIndex):
    """Redis-backed hash index for horizontally scalable, persistent search.

    Implements the same :class:`HashIndex` contract as :class:`InMemoryHashIndex`
    (and inherits the identical offset-alignment ``search``), but stores postings
    in Redis so the index survives restarts and is shareable across processes.

    Inject a ``client`` (e.g. ``fakeredis`` in tests, or a configured
    ``redis.Redis``); otherwise a client is created lazily from ``url`` so
    importing this module never requires the ``redis`` package.

    Key layout (all namespaced under ``key_prefix``)::

        <p>:files            SET  of file_id
        <p>:h:<hash_code>    LIST of "<file_id>:<time_offset>" postings
        <p>:f:<file_id>      LIST of "<hash_code>:<time_offset>" (removal + export)
        <p>:m:<file_id>      STRING json metadata
        <p>:npostings        INT  total posting count
    """

    def __init__(
        self,
        client=None,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "fpidx",
        **client_kwargs,
    ) -> None:
        self.key_prefix = key_prefix
        if client is not None:
            self._redis = client
        else:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - exercised only without redis
                raise RuntimeError(
                    "redis is required for RedisHashIndex; install it with 'pip install redis'"
                ) from exc
            client_kwargs.setdefault("decode_responses", True)
            self._redis = redis.Redis.from_url(url, **client_kwargs)

    def _key(self, *parts: str) -> str:
        return ":".join((self.key_prefix, *parts))

    @staticmethod
    def _text(value: object) -> str:
        return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)

    @property
    def file_count(self) -> int:
        return int(self._redis.scard(self._key("files")))

    @property
    def posting_count(self) -> int:
        value = self._redis.get(self._key("npostings"))
        return int(value) if value else 0

    def add(self, fingerprint: Fingerprint) -> None:
        self.remove(fingerprint.file_id)
        file_id = fingerprint.file_id
        entries = [(item.hash_code, item.time_offset) for item in fingerprint.hashes]
        metadata = {
            "file_id": file_id,
            "path": fingerprint.path,
            "handler": fingerprint.handler,
            "size_bytes": fingerprint.size_bytes,
            "content_sha256": fingerprint.content_sha256,
            "hash_count": fingerprint.hash_count,
            "landmark_count": fingerprint.landmark_count,
            **fingerprint.metadata,
        }
        pipe = self._redis.pipeline()
        pipe.sadd(self._key("files"), file_id)
        pipe.set(self._key("m", file_id), json.dumps(metadata, sort_keys=True))
        for hash_code, time_offset in entries:
            pipe.rpush(self._key("h", str(hash_code)), f"{file_id}:{time_offset}")
            pipe.rpush(self._key("f", file_id), f"{hash_code}:{time_offset}")
        if entries:
            pipe.incrby(self._key("npostings"), len(entries))
        pipe.execute()

    def remove(self, file_id: str) -> None:
        raw = self._redis.lrange(self._key("f", file_id), 0, -1)
        if not raw:
            # Clear membership/metadata even for a file that had zero postings.
            self._redis.srem(self._key("files"), file_id)
            self._redis.delete(self._key("m", file_id))
            return
        seen: set[tuple[str, str]] = set()
        pipe = self._redis.pipeline()
        for item in raw:
            hash_code, time_offset = self._text(item).rsplit(":", 1)
            if (hash_code, time_offset) in seen:
                continue
            seen.add((hash_code, time_offset))
            # count=0 removes every matching posting of this file from the hash list;
            # empty lists are auto-deleted by Redis.
            pipe.lrem(self._key("h", hash_code), 0, f"{file_id}:{time_offset}")
        pipe.delete(self._key("f", file_id))
        pipe.delete(self._key("m", file_id))
        pipe.srem(self._key("files"), file_id)
        pipe.decrby(self._key("npostings"), len(raw))
        pipe.execute()

    def query(self, hash_code: int) -> list[IndexPosting]:
        code = int(hash_code)
        raw = self._redis.lrange(self._key("h", str(code)), 0, -1)
        postings: list[IndexPosting] = []
        for item in raw:
            file_id, time_offset = self._text(item).rsplit(":", 1)
            postings.append(
                IndexPosting(file_id=file_id, hash_code=code, time_offset=int(time_offset))
            )
        return postings

    def query_many(self, hash_codes: Iterable[int]) -> dict[int, list[IndexPosting]]:
        codes = list({int(c) for c in hash_codes})
        results: dict[int, list[IndexPosting]] = {}
        if not codes:
            return results
        pipe = self._redis.pipeline()  # one round-trip for all LRANGEs
        for code in codes:
            pipe.lrange(self._key("h", str(code)), 0, -1)
        for code, raw in zip(codes, pipe.execute(), strict=False):
            postings: list[IndexPosting] = []
            for item in raw:
                file_id, time_offset = self._text(item).rsplit(":", 1)
                postings.append(
                    IndexPosting(file_id=file_id, hash_code=code, time_offset=int(time_offset))
                )
            results[code] = postings
        return results

    def _metadata_for(self, file_id: str) -> dict:
        raw = self._redis.get(self._key("m", file_id))
        if not raw:
            return {}
        try:
            data = json.loads(self._text(raw))
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def to_dict(self) -> dict[str, object]:
        files: dict[str, list[list[int]]] = {}
        metadata: dict[str, dict] = {}
        for raw_id in self._redis.smembers(self._key("files")):
            file_id = self._text(raw_id)
            entries: list[list[int]] = []
            for item in self._redis.lrange(self._key("f", file_id), 0, -1):
                hash_code, time_offset = self._text(item).rsplit(":", 1)
                entries.append([int(hash_code), int(time_offset)])
            files[file_id] = entries
            metadata[file_id] = self._metadata_for(file_id)
        return {"backend": "redis", "files": files, "metadata": metadata}


class SQLiteHashIndex(HashIndex):
    """SQLite-backed hash index: zero-dependency (stdlib), file-persistent.

    Implements the same :class:`HashIndex` contract and inherits the shared
    ``search``/``save``/``load_snapshot``. Postings live in a relational table
    indexed on ``hash_code`` (fast ``query``) and ``file_id`` (fast ``remove``).
    Pass ``":memory:"`` for an ephemeral in-process database (used by tests) or a
    file path for persistence; or inject an existing ``sqlite3.Connection``.
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
        if connection is not None:
            self._conn = connection
        else:
            self._conn = sqlite3.connect(str(database), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id  TEXT PRIMARY KEY,
                metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS postings (
                file_id     TEXT    NOT NULL,
                hash_code   INTEGER NOT NULL,
                time_offset INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_postings_hash ON postings(hash_code);
            CREATE INDEX IF NOT EXISTS idx_postings_file ON postings(file_id);
            """
        )
        self._conn.commit()

    @classmethod
    def _encode(cls, hash_code: int) -> int:
        return int(hash_code) - cls._SIGNED_OFFSET

    @classmethod
    def _decode(cls, stored: int) -> int:
        return int(stored) + cls._SIGNED_OFFSET

    @property
    def file_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    @property
    def posting_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0])

    def add(self, fingerprint: Fingerprint) -> None:
        self.remove(fingerprint.file_id)
        metadata = {
            "file_id": fingerprint.file_id,
            "path": fingerprint.path,
            "handler": fingerprint.handler,
            "size_bytes": fingerprint.size_bytes,
            "content_sha256": fingerprint.content_sha256,
            "hash_count": fingerprint.hash_count,
            "landmark_count": fingerprint.landmark_count,
            **fingerprint.metadata,
        }
        self._conn.execute(
            "INSERT INTO files (file_id, metadata) VALUES (?, ?)",
            (fingerprint.file_id, json.dumps(metadata, sort_keys=True)),
        )
        self._conn.executemany(
            "INSERT INTO postings (file_id, hash_code, time_offset) VALUES (?, ?, ?)",
            [
                (fingerprint.file_id, self._encode(item.hash_code), int(item.time_offset))
                for item in fingerprint.hashes
            ],
        )
        self._conn.commit()

    def remove(self, file_id: str) -> None:
        self._conn.execute("DELETE FROM postings WHERE file_id = ?", (file_id,))
        self._conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        self._conn.commit()

    def query(self, hash_code: int) -> list[IndexPosting]:
        code = int(hash_code)
        rows = self._conn.execute(
            "SELECT file_id, time_offset FROM postings WHERE hash_code = ?",
            (self._encode(code),),
        ).fetchall()
        return [
            IndexPosting(file_id=row[0], hash_code=code, time_offset=int(row[1]))
            for row in rows
        ]

    def query_many(self, hash_codes: Iterable[int]) -> dict[int, list[IndexPosting]]:
        codes = list({int(c) for c in hash_codes})
        results: dict[int, list[IndexPosting]] = {code: [] for code in codes}
        chunk = 500  # stay well under SQLITE_MAX_VARIABLE_NUMBER
        for start in range(0, len(codes), chunk):
            batch = [self._encode(code) for code in codes[start:start + chunk]]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT hash_code, file_id, time_offset FROM postings "
                f"WHERE hash_code IN ({placeholders})",
                batch,
            ).fetchall()
            for stored, file_id, time_offset in rows:
                code = self._decode(stored)
                results[code].append(
                    IndexPosting(file_id=file_id, hash_code=code, time_offset=int(time_offset))
                )
        return results

    def _aggregate(self, fingerprint: Fingerprint) -> dict[str, tuple[int, int, int, int]]:
        """Aggregate the offset histogram server-side via a single SQL pass.

        Loads the query's (hash_code, offset) pairs into a temp table, joins to
        the postings, and groups by (file_id, delta) -- so only per-file
        aggregates cross the boundary, not millions of postings. The winning bin
        is votes DESC then delta ASC, matching the base in-memory tie-break.
        """

        pairs = [(self._encode(h.hash_code), int(h.time_offset)) for h in fingerprint.hashes]
        if not pairs:
            return {}
        self._conn.execute("CREATE TEMP TABLE IF NOT EXISTS _query (hash_code INTEGER, qoff INTEGER)")
        self._conn.execute("DELETE FROM _query")
        self._conn.executemany("INSERT INTO _query (hash_code, qoff) VALUES (?, ?)", pairs)
        rows = self._conn.execute(
            """
            WITH matches AS (
                SELECT p.file_id AS file_id, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                FROM postings p JOIN _query q ON p.hash_code = q.hash_code
            ),
            bins AS (SELECT file_id, delta, COUNT(*) AS votes FROM matches GROUP BY file_id, delta),
            ranked AS (
                SELECT file_id, delta, votes,
                       ROW_NUMBER() OVER (PARTITION BY file_id ORDER BY votes DESC, delta ASC) AS rn
                FROM bins
            ),
            totals AS (
                SELECT file_id, COUNT(*) AS total_votes, COUNT(DISTINCT hc) AS uniq
                FROM matches GROUP BY file_id
            )
            SELECT r.file_id, r.delta, r.votes, t.total_votes, t.uniq
            FROM ranked r JOIN totals t ON r.file_id = t.file_id
            WHERE r.rn = 1
            """
        ).fetchall()
        self._conn.execute("DELETE FROM _query")
        return {
            file_id: (int(delta), int(votes), int(total), int(uniq))
            for file_id, delta, votes, total, uniq in rows
        }

    def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
        file_total = self.file_count
        if file_total == 0:
            return 0
        threshold = max_df_ratio * file_total
        cursor = self._conn.execute(
            "DELETE FROM postings WHERE hash_code IN ("
            "  SELECT hash_code FROM postings GROUP BY hash_code "
            "  HAVING COUNT(DISTINCT file_id) > ?)",
            (threshold,),
        )
        removed = cursor.rowcount
        if removed:
            counts = dict(
                self._conn.execute("SELECT file_id, COUNT(*) FROM postings GROUP BY file_id").fetchall()
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

    def to_dict(self) -> dict[str, object]:
        files: dict[str, list[list[int]]] = {}
        metadata: dict[str, dict] = {}
        for (file_id,) in self._conn.execute("SELECT file_id FROM files").fetchall():
            entries = [
                [self._decode(hash_code), int(time_offset)]
                for hash_code, time_offset in self._conn.execute(
                    "SELECT hash_code, time_offset FROM postings "
                    "WHERE file_id = ? ORDER BY rowid",
                    (file_id,),
                ).fetchall()
            ]
            files[file_id] = entries
            metadata[file_id] = self._metadata_for(file_id)
        return {"backend": "sqlite", "files": files, "metadata": metadata}

    def close(self) -> None:
        self._conn.close()


class PostgresHashIndex(HashIndex):
    """PostgreSQL-backed hash index for a shared, durable, server-grade store.

    Implements the same :class:`HashIndex` contract and inherits the shared
    ``search``/``save``/``load_snapshot``. Postings live in a table indexed on
    ``hash_code`` (fast ``query``) and ``file_id`` (fast ``remove``); metadata is
    stored as ``JSONB``. Pass a ``dsn`` (libpq connection string) or inject a
    ``psycopg`` connection. ``psycopg`` is imported lazily, so importing this
    module never requires it.

    Like SQLite, PostgreSQL ``BIGINT`` is signed 64-bit, so unsigned 64-bit hash
    codes are stored with the same reversible signed offset.
    """

    _SIGNED_OFFSET = 1 << 63

    def __init__(
        self,
        dsn: str = "postgresql://localhost/fingerprint",
        connection=None,
        table_prefix: str = "fp",
    ) -> None:
        if not table_prefix.isidentifier():
            raise ValueError("table_prefix must be a valid SQL identifier")
        self._files_table = f"{table_prefix}_files"
        self._postings_table = f"{table_prefix}_postings"
        if connection is not None:
            self._conn = connection
        else:
            try:
                import psycopg
            except ImportError as exc:  # pragma: no cover - exercised only without psycopg
                raise RuntimeError(
                    "psycopg is required for PostgresHashIndex; install it with "
                    "'pip install \"psycopg[binary]\"'"
                ) from exc
            self._conn = psycopg.connect(dsn)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._files_table} ("
                "file_id TEXT PRIMARY KEY, metadata JSONB NOT NULL)"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._postings_table} ("
                "file_id TEXT NOT NULL, hash_code BIGINT NOT NULL, time_offset INTEGER NOT NULL)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._postings_table}_hash_idx "
                f"ON {self._postings_table}(hash_code)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._postings_table}_file_idx "
                f"ON {self._postings_table}(file_id)"
            )
        self._conn.commit()

    @classmethod
    def _encode(cls, hash_code: int) -> int:
        return int(hash_code) - cls._SIGNED_OFFSET

    @classmethod
    def _decode(cls, stored: int) -> int:
        return int(stored) + cls._SIGNED_OFFSET

    @property
    def file_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._files_table}")
            return int(cur.fetchone()[0])

    @property
    def posting_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._postings_table}")
            return int(cur.fetchone()[0])

    def add(self, fingerprint: Fingerprint) -> None:
        self.remove(fingerprint.file_id)
        metadata = {
            "file_id": fingerprint.file_id,
            "path": fingerprint.path,
            "handler": fingerprint.handler,
            "size_bytes": fingerprint.size_bytes,
            "content_sha256": fingerprint.content_sha256,
            "hash_count": fingerprint.hash_count,
            "landmark_count": fingerprint.landmark_count,
            **fingerprint.metadata,
        }
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._files_table} (file_id, metadata) VALUES (%s, %s::jsonb)",
                (fingerprint.file_id, json.dumps(metadata, sort_keys=True)),
            )
            cur.executemany(
                f"INSERT INTO {self._postings_table} (file_id, hash_code, time_offset) "
                "VALUES (%s, %s, %s)",
                [
                    (fingerprint.file_id, self._encode(item.hash_code), int(item.time_offset))
                    for item in fingerprint.hashes
                ],
            )
        self._conn.commit()

    def remove(self, file_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._postings_table} WHERE file_id = %s", (file_id,))
            cur.execute(f"DELETE FROM {self._files_table} WHERE file_id = %s", (file_id,))
        self._conn.commit()

    def query(self, hash_code: int) -> list[IndexPosting]:
        code = int(hash_code)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT file_id, time_offset FROM {self._postings_table} WHERE hash_code = %s",
                (self._encode(code),),
            )
            rows = cur.fetchall()
        return [
            IndexPosting(file_id=row[0], hash_code=code, time_offset=int(row[1]))
            for row in rows
        ]

    def query_many(self, hash_codes: Iterable[int]) -> dict[int, list[IndexPosting]]:
        codes = list({int(c) for c in hash_codes})
        results: dict[int, list[IndexPosting]] = {code: [] for code in codes}
        if not codes:
            return results
        encoded = [self._encode(code) for code in codes]
        with self._conn.cursor() as cur:  # single round-trip via array membership
            cur.execute(
                f"SELECT hash_code, file_id, time_offset FROM {self._postings_table} "
                "WHERE hash_code = ANY(%s)",
                (encoded,),
            )
            for stored, file_id, time_offset in cur.fetchall():
                code = self._decode(stored)
                results[code].append(
                    IndexPosting(file_id=file_id, hash_code=code, time_offset=int(time_offset))
                )
        return results

    def _aggregate(self, fingerprint: Fingerprint) -> dict[str, tuple[int, int, int, int]]:
        """Aggregate the offset histogram server-side in one round-trip.

        The query's (hash_code, offset) pairs are passed as two arrays and
        unnested into a derived table, joined to the postings, then grouped by
        (file_id, delta). Only per-file aggregates return. Winning bin is votes
        DESC then delta ASC, matching the base in-memory tie-break.
        """

        if not fingerprint.hashes:
            return {}
        codes = [self._encode(h.hash_code) for h in fingerprint.hashes]
        offsets = [int(h.time_offset) for h in fingerprint.hashes]
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                WITH q(hash_code, qoff) AS (SELECT * FROM unnest(%s::bigint[], %s::int[])),
                matches AS (
                    SELECT p.file_id AS file_id, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                    FROM {self._postings_table} p JOIN q ON p.hash_code = q.hash_code
                ),
                bins AS (SELECT file_id, delta, COUNT(*) AS votes FROM matches GROUP BY file_id, delta),
                ranked AS (
                    SELECT file_id, delta, votes,
                           ROW_NUMBER() OVER (PARTITION BY file_id ORDER BY votes DESC, delta ASC) AS rn
                    FROM bins
                ),
                totals AS (
                    SELECT file_id, COUNT(*) AS total_votes, COUNT(DISTINCT hc) AS uniq
                    FROM matches GROUP BY file_id
                )
                SELECT r.file_id, r.delta, r.votes, t.total_votes, t.uniq
                FROM ranked r JOIN totals t ON r.file_id = t.file_id
                WHERE r.rn = 1
                """,
                (codes, offsets),
            )
            rows = cur.fetchall()
        return {
            file_id: (int(delta), int(votes), int(total), int(uniq))
            for file_id, delta, votes, total, uniq in rows
        }

    def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
        file_total = self.file_count
        if file_total == 0:
            return 0
        threshold = max_df_ratio * file_total
        posts, files = self._postings_table, self._files_table
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {posts} WHERE hash_code IN ("
                f"  SELECT hash_code FROM {posts} GROUP BY hash_code "
                f"  HAVING COUNT(DISTINCT file_id) > %s)",
                (threshold,),
            )
            removed = cur.rowcount
            if removed:
                # Recalibrate stored hash_count to remaining postings per file.
                cur.execute(
                    f"UPDATE {files} f SET metadata = "
                    f"jsonb_set(f.metadata, '{{hash_count}}', to_jsonb(c.cnt)) "
                    f"FROM (SELECT file_id, COUNT(*) AS cnt FROM {posts} GROUP BY file_id) c "
                    f"WHERE f.file_id = c.file_id"
                )
                cur.execute(
                    f"UPDATE {files} SET metadata = jsonb_set(metadata, '{{hash_count}}', '0'::jsonb) "
                    f"WHERE file_id NOT IN (SELECT DISTINCT file_id FROM {posts})"
                )
        self._conn.commit()
        return removed

    def _metadata_for(self, file_id: str) -> dict:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT metadata FROM {self._files_table} WHERE file_id = %s", (file_id,)
            )
            row = cur.fetchone()
        if not row:
            return {}
        data = row[0]
        if isinstance(data, str):  # some drivers return JSONB as text
            try:
                data = json.loads(data)
            except (ValueError, TypeError):
                return {}
        return data if isinstance(data, dict) else {}

    def to_dict(self) -> dict[str, object]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT file_id FROM {self._files_table}")
            file_ids = [row[0] for row in cur.fetchall()]
        files: dict[str, list[list[int]]] = {}
        metadata: dict[str, dict] = {}
        for file_id in file_ids:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT hash_code, time_offset FROM {self._postings_table} "
                    "WHERE file_id = %s ORDER BY hash_code, time_offset",
                    (file_id,),
                )
                files[file_id] = [
                    [self._decode(hash_code), int(time_offset)]
                    for hash_code, time_offset in cur.fetchall()
                ]
            metadata[file_id] = self._metadata_for(file_id)
        return {"backend": "postgres", "files": files, "metadata": metadata}

    def close(self) -> None:
        self._conn.close()
