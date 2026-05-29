"""Hash index interfaces and the default in-memory backend."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict

from .models import ConstellationHash, Fingerprint, IndexPosting, SearchResult


class HashIndex(ABC):
    """Storage-agnostic contract for searchable fingerprint postings."""

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
    def save(self, path: str | Path) -> None:
        """Persist the index."""

    def search(self, fingerprint: Fingerprint, top_k: int = 10) -> list[SearchResult]:
        """Return ranked matches via Shazam-style offset-histogram alignment.

        Backend-agnostic: it relies only on ``query()`` and ``_metadata_for()``,
        so every backend ranks identically and scores stay comparable.
        """

        offset_histograms: dict[str, Counter[int]] = defaultdict(Counter)
        total_votes: Counter[str] = Counter()
        unique_hashes: dict[str, set[int]] = defaultdict(set)

        for query_hash in fingerprint.hashes:
            for posting in self.query(query_hash.hash_code):
                offset = posting.time_offset - query_hash.time_offset
                offset_histograms[posting.file_id][offset] += 1
                total_votes[posting.file_id] += 1
                unique_hashes[posting.file_id].add(query_hash.hash_code)

        results: list[SearchResult] = []
        query_hash_count = max(1, fingerprint.hash_count)
        for file_id, histogram in offset_histograms.items():
            if not histogram:
                continue
            offset, aligned_votes = histogram.most_common(1)[0]
            total = total_votes[file_id]
            unique = len(unique_hashes[file_id])
            alignment_ratio = aligned_votes / max(1, total)
            coverage_ratio = unique / query_hash_count
            score = aligned_votes + (0.30 * unique) + (5.0 * alignment_ratio) + (2.0 * coverage_ratio)
            results.append(
                SearchResult(
                    file_id=file_id,
                    score=round(float(score), 6),
                    aligned_votes=int(aligned_votes),
                    total_votes=int(total),
                    unique_hashes=int(unique),
                    offset=int(offset),
                    metadata=dict(self._metadata_for(file_id)),
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


class InMemoryHashIndex(HashIndex):
    """Dict-backed hash index with Shazam-style offset alignment scoring."""

    def __init__(self) -> None:
        self._postings: DefaultDict[int, list[IndexPosting]] = defaultdict(list)
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

    def _metadata_for(self, file_id: str) -> dict:
        return dict(self._metadata.get(file_id, {}))

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": "in_memory",
            "files": self._file_entries,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "InMemoryHashIndex":
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

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, sort_keys=True, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryHashIndex":
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

    def save(self, path: str | Path) -> None:
        """Export a portable JSON snapshot (same schema as InMemoryHashIndex)."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, sort_keys=True, separators=(",", ":"))

    def load_snapshot(self, path: str | Path) -> "RedisHashIndex":
        """Bulk-load a JSON snapshot (from any backend's ``save``) into Redis."""

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
                    metadata={k: v for k, v in meta.items()
                              if k not in {"file_id", "path", "handler", "size_bytes",
                                           "content_sha256", "hash_count", "landmark_count"}},
                )
            )
        return self
