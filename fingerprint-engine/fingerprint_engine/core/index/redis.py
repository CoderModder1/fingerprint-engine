"""Redis-backed hash index."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable

from ..models import (
    Fingerprint,
    IndexPosting,
)
from ._common import (
    _index_metadata,
    _synchronized,
)
from .base import HashIndex

logger = logging.getLogger(__name__)


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
        <p>:format_version   STRING corpus hash-format version (only if non-default)

    Concurrency: a ``redis-py`` client and its pipelines are thread-safe at the
    command level, but :meth:`add`/:meth:`add_many`/:meth:`remove` are
    multi-step read-modify-write sequences (read a file's existing postings,
    then issue the dependent removals/inserts and adjust the ``npostings``
    counter). Under the shared FastAPI-threadpool index two such sequences on
    the same ``file_id`` would interleave -- double-counting postings and the
    counter and skewing search confidence -- so the mutators are serialized by a
    per-index :class:`threading.RLock` (the ``@_synchronized`` decorator), the
    same in-process contract the SQL backends use. (Cross-process atomicity for
    multiple service instances on one real Redis is a separate, deferred concern
    -- a per-process lock cannot cover it; see SECURITY.md.)
    """

    def __init__(
        self,
        client=None,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "fpidx",
        **client_kwargs,
    ) -> None:
        # Serializes the read-modify-write mutators (add/add_many/remove) within
        # one process. RLock (not Lock) because add() calls remove() and both are
        # @_synchronized; uncontended on the single-threaded path so output is
        # unchanged. See the class docstring for the cross-process caveat.
        self._lock = threading.RLock()
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
        # F2: restore the corpus hash-format version from its durable key so a
        # reopen against the same store reports the right version instead of
        # silently defaulting to baseline. Absent (a fresh store, or one built by
        # an older version / a default corpus) -> stay at the unpinned default.
        raw = self._redis.get(self._key("format_version"))
        if raw is not None:
            self._load_persisted_format_version(self._text(raw))

    def _persist_format_version(self, version: int) -> None:
        # F2: stamp the pinned (non-default) version under a dedicated key so a
        # reopen restores it. Only ever called once, on first pin of a non-default
        # corpus, so the default path never writes this key.
        self._redis.set(self._key("format_version"), str(version))

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

    @_synchronized
    def add(self, fingerprint: Fingerprint) -> None:
        self._record_format_version(fingerprint)
        self.remove(fingerprint.file_id)
        file_id = fingerprint.file_id
        entries = [(item.hash_code, item.time_offset) for item in fingerprint.hashes]
        metadata = _index_metadata(fingerprint)
        pipe = self._redis.pipeline()
        pipe.sadd(self._key("files"), file_id)
        pipe.set(self._key("m", file_id), json.dumps(metadata, sort_keys=True))
        for hash_code, time_offset in entries:
            pipe.rpush(self._key("h", str(hash_code)), f"{file_id}:{time_offset}")
            pipe.rpush(self._key("f", file_id), f"{hash_code}:{time_offset}")
        if entries:
            pipe.incrby(self._key("npostings"), len(entries))
        pipe.execute()

    @_synchronized
    def add_many(self, fingerprints: Iterable[Fingerprint]) -> None:
        """Batch the whole ingest into one read + one write pipeline.

        Equivalent to per-item :meth:`add`. A sequential ``add()`` of a duplicate
        ``file_id`` removes the earlier copy (its just-inserted postings) before
        inserting the later one, so the last fingerprint for each ``file_id``
        wins and the file's *pre-batch* postings are removed exactly once. We
        replicate that directly: keep the last fingerprint per ``file_id`` (in
        first-seen order is irrelevant -- only the survivor matters for state),
        read each file's existing postings once, then issue all removals and
        inserts in a single pipeline so the batch costs two round-trips, not two
        per file.
        """

        # Fail-closed PRE-CHECK (no pin/persist) so a cross-format batch raises
        # before any write OR version stamp -- true all-or-nothing (F1).
        fingerprints = list(fingerprints)
        self._validate_batch_format(fingerprints)
        # Collapse to the surviving (last) fingerprint per file_id, preserving
        # the order in which each file_id first appeared for deterministic ops.
        survivors: dict[str, Fingerprint] = {}
        for fingerprint in fingerprints:
            self._record_format_version(fingerprint)
            survivors[fingerprint.file_id] = fingerprint
        if not survivors:
            return

        # One pipeline to read every target file's existing postings (drives the
        # remove() decision without a per-file round-trip).
        read_pipe = self._redis.pipeline()
        for file_id in survivors:
            read_pipe.lrange(self._key("f", file_id), 0, -1)
        existing_raw = read_pipe.execute()

        pipe = self._redis.pipeline()
        net_delta = 0
        for (file_id, fingerprint), raw in zip(survivors.items(), existing_raw, strict=True):
            # --- remove() effect for the pre-batch state of this file_id ---
            if raw:
                seen: set[tuple[str, str]] = set()
                for item in raw:
                    rem_code, rem_off = self._text(item).rsplit(":", 1)
                    if (rem_code, rem_off) in seen:
                        continue
                    seen.add((rem_code, rem_off))
                    pipe.lrem(self._key("h", rem_code), 0, f"{file_id}:{rem_off}")
                pipe.delete(self._key("f", file_id))
                net_delta -= len(raw)
            pipe.delete(self._key("m", file_id))
            # srem is harmless for a not-yet-member id; sadd below re-adds it.
            pipe.srem(self._key("files"), file_id)

            # --- add() effect for the surviving fingerprint ---
            entries = [(item.hash_code, item.time_offset) for item in fingerprint.hashes]
            metadata = _index_metadata(fingerprint)
            pipe.sadd(self._key("files"), file_id)
            pipe.set(self._key("m", file_id), json.dumps(metadata, sort_keys=True))
            for hash_code, time_offset in entries:
                pipe.rpush(self._key("h", str(hash_code)), f"{file_id}:{time_offset}")
                pipe.rpush(self._key("f", file_id), f"{hash_code}:{time_offset}")
            net_delta += len(entries)

        if net_delta:
            pipe.incrby(self._key("npostings"), net_delta)
        pipe.execute()

    @_synchronized
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

    def list_files(self) -> list[str]:
        return sorted(self._text(raw_id) for raw_id in self._redis.smembers(self._key("files")))

    def contains(self, file_id: str) -> bool:
        return bool(self._redis.sismember(self._key("files"), file_id))

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
