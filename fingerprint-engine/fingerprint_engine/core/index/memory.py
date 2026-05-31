"""In-memory (dict-backed, JSON-persisted) hash index -- the default backend."""

from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from ..exceptions import (
    InvalidSnapshotError,
)
from ..models import (
    Fingerprint,
    IndexPosting,
)
from ._common import (
    _in_hash_range,
    _index_metadata,
    _snapshot_format_version,
    _validate_schema_version,
)
from .base import HashIndex

logger = logging.getLogger(__name__)


class InMemoryHashIndex(HashIndex):
    """Dict-backed hash index with Shazam-style offset alignment scoring.

    Concurrency contract: reads (:meth:`query`, :meth:`query_many`,
    :meth:`search`, :meth:`_metadata_for`) are GIL-safe and lock-free. The
    mutating methods (:meth:`add`, :meth:`remove`, :meth:`prune_stop_hashes`)
    are serialized by a single re-entrant lock, so concurrent writers do not
    interleave and a concurrent reader can never observe a half-applied
    add/remove. This is a single logical writer model: parallel writes are
    safe but run one at a time.

    Storage layout (internal, output-preserving): the 64-char SHA-256
    ``file_id`` is the dominant per-posting cost when stored verbatim on every
    posting. Instead each ``file_id`` is interned to a compact non-negative
    integer surrogate (``_intern_file_id``) and the posting lists store
    ``(file_surrogate, time_offset)`` int pairs rather than full
    :class:`IndexPosting` objects holding a 64-char string reference. The
    surrogate is mapped back to the original ``file_id`` string at the query
    boundary (:meth:`query` / :meth:`query_many` rebuild real
    :class:`IndexPosting` objects) and at the aggregation boundary
    (:meth:`_aggregate` keys its result by the original ``file_id``), so
    :meth:`search`, :meth:`to_dict`, :meth:`list_files`, :meth:`iter_metadata`,
    :meth:`query`, :meth:`query_many`, and cross-backend parity are all
    BYTE-IDENTICAL to storing the string verbatim. This is a pure footprint
    optimization with no observable effect: ``_file_entries`` / ``_metadata``
    remain keyed by the original ``file_id`` string (one entry per file, not per
    posting -- already compact, and returned directly by the snapshot methods).

    Surrogate allocation is monotonic and append-only within a process: a
    brand-new ``file_id`` gets a fresh surrogate, and a re-added ``file_id``
    REUSES its existing surrogate (so re-indexing the same file never grows the
    maps). Crucially, :meth:`remove` does NOT delete a file's surrogate
    mappings -- it leaves ``_id_to_fid`` / ``_fid_to_id`` intact -- so a
    lock-free reader that has already read a posting referencing that surrogate
    can still dereference it to the correct ``file_id`` instead of racing a
    concurrent ``del`` and raising ``KeyError``. This upholds the read contract
    above: a concurrent reader can never observe a half-applied add/remove.

    The deliberate price of lock-free reads is that surrogates for files which
    are removed and NEVER re-added are retained: ``_id_to_fid`` / ``_fid_to_id``
    form a bounded, monotonic set whose size is the total number of DISTINCT
    ``file_id`` ever seen (not the live file count). This is harmless for
    correctness -- a retired surrogate has no postings, so it never appears in
    any query/search/aggregate output -- and bounded by the workload's distinct
    id count. A future compaction could reclaim retired surrogates under the
    write lock (once no reader can hold a posting referencing them); it is NOT
    implemented here. Because surrogates are an internal detail never exposed,
    their concrete values are not part of any contract.
    """

    def __init__(self) -> None:
        # Posting lists store (file_surrogate:int, time_offset:int) pairs; the
        # surrogate is mapped back to the str file_id only at the query/aggregate
        # boundary. Storing the small int per posting (instead of an IndexPosting
        # carrying a 64-char file_id string) is the footprint win.
        self._postings: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        self._file_entries: dict[str, list[tuple[int, int]]] = {}
        self._metadata: dict[str, dict[str, object]] = {}
        # Bidirectional file_id <-> integer surrogate maps. ``_id_to_fid`` maps a
        # surrogate to its str file_id; ``_fid_to_id`` is the inverse. These are
        # APPEND-ONLY: a surrogate is allocated once per distinct file_id and is
        # never deleted (not even on remove()) and never reused, so a surrogate
        # uniquely and permanently identifies one file_id for the lifetime of the
        # index. remove() deliberately leaves these intact (see class docstring)
        # so a lock-free reader can always resolve a just-removed posting's
        # surrogate instead of racing a concurrent del and raising KeyError.
        self._id_to_fid: dict[int, str] = {}
        self._fid_to_id: dict[str, int] = {}
        self._next_surrogate = 0
        # Re-entrant so remove() called from within add() under the same lock
        # does not deadlock.
        self._write_lock = threading.RLock()

    def _intern_file_id(self, file_id: str) -> int:
        """Return the stable integer surrogate for ``file_id``.

        Called only under :attr:`_write_lock` from :meth:`add` /
        :meth:`from_dict`. Surrogates are append-only and stable per ``file_id``:
        if this ``file_id`` already has a surrogate (it was added before, then
        possibly removed -- :meth:`remove` keeps the mapping), that SAME surrogate
        is reused, so re-indexing a file never grows the maps. Only a brand-new
        ``file_id`` allocates a fresh monotonic surrogate. A surrogate is never
        reused for a different ``file_id``, so a stale surrogate can never
        silently alias another file.
        """

        existing = self._fid_to_id.get(file_id)
        if existing is not None:
            return existing
        surrogate = self._next_surrogate
        self._next_surrogate += 1
        self._id_to_fid[surrogate] = file_id
        self._fid_to_id[file_id] = surrogate
        return surrogate

    @property
    def file_count(self) -> int:
        return len(self._file_entries)

    @property
    def posting_count(self) -> int:
        return sum(len(postings) for postings in self._postings.values())

    def add(self, fingerprint: Fingerprint) -> None:
        with self._write_lock:
            self._record_format_version(fingerprint)
            self.remove(fingerprint.file_id)
            entries = [(item.hash_code, item.time_offset) for item in fingerprint.hashes]
            self._file_entries[fingerprint.file_id] = entries
            self._metadata[fingerprint.file_id] = _index_metadata(fingerprint)
            surrogate = self._intern_file_id(fingerprint.file_id)
            for hash_code, time_offset in entries:
                # Store the compact (surrogate, time_offset) int pair; the str
                # file_id is recovered from the surrogate at the query boundary.
                self._postings[hash_code].append((surrogate, time_offset))

    def add_many(self, fingerprints: Iterable[Fingerprint]) -> None:
        # Hold the write lock once for the whole batch (re-entrant, so the
        # per-item add() re-acquire is free) so a concurrent reader never
        # observes the batch half-applied; the per-add() effect is otherwise
        # identical to the sequential default.
        with self._write_lock:
            materialized = list(fingerprints)
            # Fail-closed (F1): validate the WHOLE batch up front WITHOUT pinning
            # or persisting, so a cross-format batch raises BEFORE any mutation and
            # leaves the index (and its version) untouched -- true all-or-nothing,
            # like the single-transaction SQL backends. A no-op under
            # allow_format_mixing, where per-item add() warns and proceeds instead.
            self._validate_batch_format(materialized)
            for fingerprint in materialized:
                self.add(fingerprint)

    def remove(self, file_id: str) -> None:
        with self._write_lock:
            if file_id not in self._file_entries:
                return

            surrogate = self._fid_to_id[file_id]
            for hash_code, _time_offset in self._file_entries[file_id]:
                self._postings[hash_code] = [
                    posting
                    for posting in self._postings[hash_code]
                    if posting[0] != surrogate
                ]
                if not self._postings[hash_code]:
                    del self._postings[hash_code]

            del self._file_entries[file_id]
            self._metadata.pop(file_id, None)
            # Deliberately DO NOT delete the surrogate mappings. A lock-free
            # reader (query()/_aggregate()) may have already read a posting that
            # references this surrogate and is about to dereference it via
            # _id_to_fid; deleting the entry here would race that read into a
            # KeyError. Leaving the (postings-less) surrogate in place keeps the
            # read contract -- a reader can never observe a half-applied remove --
            # and lets a re-add of this file_id reuse the SAME surrogate. The
            # retained mappings are bounded/monotonic (one per distinct file_id
            # ever seen); reclaiming them is left to a future compaction pass.

    def query(self, hash_code: int) -> list[IndexPosting]:
        # Map each stored (surrogate, time_offset) pair back to a real
        # IndexPosting carrying the original str file_id -- the public return
        # type and values are identical to storing IndexPosting verbatim. The
        # list is rebuilt in stored order, so query() ordering is unchanged.
        code = int(hash_code)
        id_to_fid = self._id_to_fid
        return [
            IndexPosting(
                file_id=id_to_fid[surrogate],
                hash_code=code,
                time_offset=time_offset,
            )
            for surrogate, time_offset in self._postings.get(code, ())
        ]

    def _aggregate(
        self,
        fingerprint: Fingerprint,
        offset_tolerance: int = 0,
        candidates: set[str] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Surrogate-keyed in-memory aggregation, output-identical to the base.

        Equivalent to :meth:`HashIndex._aggregate` but accumulates the per-file
        offset histogram / vote tallies keyed by the compact integer surrogate
        (read straight off the stored ``(surrogate, time_offset)`` posting pairs)
        instead of the 64-char ``file_id`` string, then maps each surviving
        surrogate back to its ``file_id`` only when building the final result
        dict. This avoids both materializing an :class:`IndexPosting` per posting
        and hashing the long ``file_id`` string in the hot loop, while producing
        the IDENTICAL per-file ``(offset, aligned_votes, total_votes,
        unique_hashes)`` aggregates the base produces -- so :meth:`search`
        rankings are byte-identical.

        Iteration matches the base exactly: query hashes are walked in
        ``fingerprint.hashes`` order (duplicates included, so ``total_votes``
        counts repeats), and for each its posting list is read directly from
        ``self._postings``. The ``candidates`` set (of ``file_id`` strings) is
        translated once to a surrogate set so the per-posting membership test is
        an int comparison; a non-candidate ``file_id`` (or one with no live
        surrogate) is skipped, exactly as the base skips a non-candidate posting.
        """

        postings = self._postings
        id_to_fid = self._id_to_fid

        candidate_surrogates: set[int] | None = None
        if candidates is not None:
            fid_to_id = self._fid_to_id
            candidate_surrogates = {
                fid_to_id[fid] for fid in candidates if fid in fid_to_id
            }

        offset_histograms: dict[int, Counter[int]] = defaultdict(Counter)
        total_votes: Counter[int] = Counter()
        unique_hashes: dict[int, set[int]] = defaultdict(set)

        for query_hash in fingerprint.hashes:
            hash_code = query_hash.hash_code
            query_offset = query_hash.time_offset
            for surrogate, time_offset in postings.get(hash_code, ()):
                if candidate_surrogates is not None and surrogate not in candidate_surrogates:
                    continue
                offset = time_offset - query_offset
                offset_histograms[surrogate][offset] += 1
                total_votes[surrogate] += 1
                unique_hashes[surrogate].add(hash_code)

        aggregates: dict[str, tuple[int, int, int, int]] = {}
        for surrogate, histogram in offset_histograms.items():
            if not histogram:
                continue
            offset, aligned = self._banded_winner(histogram, offset_tolerance)
            aggregates[id_to_fid[surrogate]] = (
                offset,
                aligned,
                total_votes[surrogate],
                len(unique_hashes[surrogate]),
            )
        return aggregates

    def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
        with self._write_lock:
            file_total = len(self._file_entries)
            if file_total == 0:
                return 0
            threshold = max_df_ratio * file_total
            # Distinct files touching a code == distinct surrogates in its
            # posting list (each surrogate maps 1:1 to a file_id), so this is
            # the same document-frequency test as counting distinct file_ids.
            stop = {
                code
                for code, postings in self._postings.items()
                if len({posting[0] for posting in postings}) > threshold
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

    def list_files(self) -> list[str]:
        return sorted(self._file_entries)

    def contains(self, file_id: str) -> bool:
        return file_id in self._file_entries

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": "in_memory",
            "files": self._file_entries,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> InMemoryHashIndex:
        index = cls()
        _validate_schema_version(data)
        # Restore the hash-derivation format version (default for a legacy/absent
        # field) so a search against the rebuilt index detects a cross-format
        # query. from_dict bypasses add(), so pin it explicitly.
        index._format_version = _snapshot_format_version(data)
        index._format_version_pinned = True
        files = data.get("files", {})
        if not isinstance(files, dict):
            raise InvalidSnapshotError("invalid index: files must be a mapping")
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            index._metadata = {
                str(file_id): dict(value) if isinstance(value, dict) else {}
                for file_id, value in metadata.items()
            }

        total_dropped = 0
        for file_id, entries in files.items():
            normalized_entries: list[tuple[int, int]] = []
            if not isinstance(entries, list):
                continue
            # Allocate the surrogate once per file, then store compact
            # (surrogate, time_offset) pairs -- identical postings to building
            # IndexPosting objects, but with the str file_id stored once.
            surrogate = index._intern_file_id(str(file_id))
            for entry in entries:
                if not isinstance(entry, list | tuple) or len(entry) != 2:
                    continue
                hash_code = int(entry[0])
                time_offset = int(entry[1])
                if not _in_hash_range(hash_code):
                    continue
                normalized_entries.append((hash_code, time_offset))
                index._postings[hash_code].append((surrogate, time_offset))
            index._file_entries[str(file_id)] = normalized_entries
            # Unlike load_snapshot (which rebuilds via add()), from_dict copies the
            # snapshot metadata wholesale, so a dropped posting would leave the
            # stored hash_count STALE -- inflating the confidence denominator and
            # deflating every match's confidence for that file. Recompute it to the
            # postings actually loaded, and surface the drop. Only touched when a
            # drop occurred, so a clean snapshot is byte-identical.
            dropped = len(entries) - len(normalized_entries)
            if dropped:
                total_dropped += dropped
                meta = index._metadata.setdefault(str(file_id), {})
                meta["hash_count"] = len(normalized_entries)
                logger.warning(
                    "snapshot file %s: skipped %d malformed/out-of-range posting(s); "
                    "recomputed hash_count to %d",
                    file_id,
                    dropped,
                    len(normalized_entries),
                )
        if total_dropped:
            logger.warning(
                "loaded index with %d posting(s) skipped as malformed or "
                "out-of-range; the index is degraded -- re-fingerprint the source "
                "to restore full recall",
                total_dropped,
            )
        return index

    @classmethod
    def load(cls, path: str | Path) -> InMemoryHashIndex:
        data = cls._read_snapshot(path)
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise InvalidSnapshotError("invalid index file")
        # from_dict validates schema_version and the files mapping.
        return cls.from_dict(data)
