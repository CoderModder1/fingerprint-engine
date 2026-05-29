"""Hash index interfaces and the default in-memory backend."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict

from .models import Fingerprint, IndexPosting, SearchResult


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
    def search(self, fingerprint: Fingerprint, top_k: int = 10) -> list[SearchResult]:
        """Return ranked matches for a query fingerprint."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the index."""


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

    def search(self, fingerprint: Fingerprint, top_k: int = 10) -> list[SearchResult]:
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
                    metadata=dict(self._metadata.get(file_id, {})),
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
