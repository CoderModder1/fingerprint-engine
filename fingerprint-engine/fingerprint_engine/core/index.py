"""Hash index interfaces and the default in-memory backend."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .exceptions import InvalidSnapshotError
from .models import Calibration, ConstellationHash, Fingerprint, IndexPosting, SearchResult

logger = logging.getLogger(__name__)

# Schema version stamped into every snapshot written by :meth:`HashIndex.save`.
# A snapshot whose top-level ``schema_version`` is present but not in
# ``_SUPPORTED_SCHEMA_VERSIONS`` is rejected on load; an ABSENT version is
# treated as version 1 for backward compatibility with already-written files.
SNAPSHOT_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

# Hash codes are unsigned 64-bit; reject snapshot postings outside this range
# before they reach a SQL backend's signed-offset encode (which would raise
# OverflowError deep in the load and abort the whole cross-backend import).
_HASH_CODE_MIN = 0
_HASH_CODE_MAX = (1 << 64) - 1


def _validate_schema_version(data: dict[str, object]) -> None:
    """Raise :class:`InvalidSnapshotError` for an unsupported snapshot version.

    An absent ``schema_version`` is treated as version 1 (legacy snapshots
    written before versioning was introduced remain loadable).
    """

    if "schema_version" not in data:
        return
    version = data["schema_version"]
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise InvalidSnapshotError(
            f"unsupported snapshot schema_version {version!r}; "
            f"this build supports {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )


def _in_hash_range(hash_code: int) -> bool:
    """Whether ``hash_code`` fits the unsigned 64-bit posting range."""

    return _HASH_CODE_MIN <= hash_code <= _HASH_CODE_MAX


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

    def add_many(self, fingerprints: Iterable[Fingerprint]) -> None:
        """Add or replace many fingerprints, equivalently to per-item :meth:`add`.

        The default fans out to :meth:`add` in order, so the result is exactly as
        if each fingerprint were added sequentially -- same remove-then-insert
        replace semantics (a later fingerprint sharing an earlier ``file_id`` wins),
        same postings, same metadata, same search results. Storage backends SHOULD
        override this to batch the writes into a single transaction/round-trip
        (per-:meth:`add` commits are fsync-bound), but the *observable* outcome MUST
        stay identical to this sequential default.
        """

        for fingerprint in fingerprints:
            self.add(fingerprint)

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

    @abstractmethod
    def list_files(self) -> list[str]:
        """Return every indexed ``file_id``, sorted for deterministic output.

        Cheaper than :meth:`to_dict` (which materializes all postings): backends
        enumerate the file-id set directly. Sorted ascending so the order is
        identical across backends regardless of their internal storage order.
        """

    def iter_metadata(self) -> Iterator[dict]:
        """Yield each file's stored metadata dict, in :meth:`list_files` order.

        Streams one metadata dict per file (the same shape :meth:`_metadata_for`
        returns: ``file_id``, ``path``, ``handler``, ``size_bytes``,
        ``content_sha256``, ``hash_count``, ``landmark_count``, plus any extra
        per-file metadata) without building the heavy whole-index
        :meth:`to_dict`. The default composes :meth:`list_files` with
        :meth:`_metadata_for`, so it is parity-identical across every backend;
        a backend MAY override it only if it can stream the same dicts, in the
        same order, more efficiently.
        """

        for file_id in self.list_files():
            yield self._metadata_for(file_id)

    def contains(self, file_id: str) -> bool:
        """Whether ``file_id`` is indexed (membership without loading postings).

        Used by incremental ingest to skip already-indexed files. The default
        consults :meth:`_metadata_for` (present iff the file was added); backends
        SHOULD override with an O(1) membership check (a key/row existence probe)
        rather than fetching and decoding the metadata blob.
        """

        return bool(self._metadata_for(file_id))

    def __contains__(self, file_id: object) -> bool:
        """``file_id in index`` membership, delegating to :meth:`contains`."""

        return isinstance(file_id, str) and self.contains(file_id)

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
        offset_tolerance: int | None = None,
        candidate_limit: int | None = None,
    ) -> list[SearchResult]:
        """Return ranked matches via Shazam-style offset-histogram alignment.

        Aggregation (per file: winning offset, aligned/total votes, unique
        hashes) is delegated to :meth:`_aggregate` so SQL backends can compute it
        server-side; scoring/calibration/ranking is shared here, so every backend
        produces identical results. Each result carries a handler-independent
        ``confidence`` in [0, 1] (aligned votes / the smaller fingerprint's hash
        count); when a :class:`Calibration` is supplied, results below its
        per-handler threshold are dropped.

        ``offset_tolerance`` is OPT-IN and DEFAULT-OFF. When it resolves to ``0``
        (the default -- left ``None`` with no calibration, or a
        :class:`Calibration` whose ``offset_tolerance`` is ``0``) the winning
        offset bin is the single exact-delta histogram peak and the returned
        rankings/scores/offsets are BYTE-IDENTICAL to behaviour before this
        option existed. When it resolves to ``> 0`` the winning bin's
        ``aligned_votes`` sums the votes of every delta bin within
        ``+-offset_tolerance`` of a candidate centre delta; the centre that
        maximises that banded sum wins (ties broken by the SMALLER centre delta,
        matching the exact-bin tie-break). Banding recovers recall on multi-edit
        near-duplicates whose aligned votes otherwise fragment across adjacent
        delta bins (each inserted/deleted run shifts the absolute frame index of
        everything after it). An explicit argument here overrides the
        calibration's field.

        ``candidate_limit`` is OPT-IN and DEFAULT-OFF (``None``). When ``None``
        (the default) the FULL exact search runs: every file sharing any query
        hash is offset-voted, and the returned rankings/scores/offsets are
        BYTE-IDENTICAL to behaviour before this option existed -- the prefilter
        code below is never entered. When set to a positive integer it caps the
        cost of large-corpus search with a cheap, sublinear-in-corpus candidate
        prefilter: files are first ranked by SHARED-HASH COUNT (how many distinct
        query hash codes touch each file -- a single batched posting lookup, no
        per-posting offset arithmetic), and only the top ``candidate_limit`` files
        proceed to the exact offset-histogram aggregation and scoring.

        Exactness/recall trade-off: shared-hash count is a monotone UPPER BOUND on
        a file's aligned votes (a hash can only vote at the winning offset if it
        is shared at all), so for a high-overlap match -- a self-match, a
        near-duplicate -- the true file is guaranteed to be among the highest
        shared-hash-count files and thus in the candidate set. When the candidate
        set is a SUPERSET of the true top-``top_k`` (the normal case for a
        generous ``candidate_limit``), the final ranking is IDENTICAL to full
        search: scoring runs unchanged on the surviving files and the dropped
        files could never have out-ranked them. The only files a too-tight limit
        can drop are *low*-overlap ones -- weak partial matches that share few
        hashes but happen to align coherently -- so set ``candidate_limit``
        comfortably above ``top_k`` to keep recall at 1.0 for genuine matches.
        Ties in shared-hash count at the cut boundary are broken deterministically
        (count DESC, then file_id ASC) so the candidate set is reproducible.
        """

        resolved_tolerance = self._resolve_offset_tolerance(offset_tolerance, calibration)
        candidates = self._select_candidates(fingerprint, candidate_limit)
        started = time.perf_counter()
        aggregates = self._aggregate(fingerprint, resolved_tolerance, candidates)
        results = self._finalize(fingerprint, aggregates, top_k, calibration)
        logger.debug(
            "search: %d query hashes -> %d candidates -> %d results in %.3f ms",
            fingerprint.hash_count,
            len(aggregates),
            len(results),
            (time.perf_counter() - started) * 1000.0,
        )
        return results

    def _select_candidates(
        self, fingerprint: Fingerprint, candidate_limit: int | None
    ) -> set[str] | None:
        """Cheap shared-hash candidate prefilter; ``None`` (default) = OFF.

        Returns ``None`` when ``candidate_limit`` is ``None`` (the default) so the
        caller runs the full exact search on every file -- the byte-identical
        legacy path. When ``candidate_limit`` is a non-negative int, returns the
        set of the top-``candidate_limit`` ``file_id`` ranked by SHARED-HASH COUNT
        (the number of distinct query hash codes whose posting list contains the
        file). This count is computed from the SAME batched :meth:`query_many`
        fetch the exact aggregation would do, but without any per-posting offset
        arithmetic, and -- crucially -- is a monotone upper bound on a file's
        aligned votes, so a high-overlap match (self / near-dup) is always
        retained. The cut is deterministic: shared-hash count DESC, then file_id
        ASC, so a boundary tie is resolved reproducibly.

        ``candidate_limit == 0`` selects no files (an explicit empty search). A
        negative limit is rejected so the cut is always well defined.
        """

        if candidate_limit is None:
            return None
        if candidate_limit < 0:
            raise ValueError("candidate_limit must be non-negative or None (None = off)")
        shared: Counter[str] = Counter()
        postings_by_code = self.query_many({qh.hash_code for qh in fingerprint.hashes})
        for postings in postings_by_code.values():
            # One vote per (file) per distinct query code: a code present in the
            # query that hits this file contributes exactly one to its shared
            # count, regardless of how many times it occurs in the file.
            for file_id in {posting.file_id for posting in postings}:
                shared[file_id] += 1
        if len(shared) <= candidate_limit:
            return set(shared)
        # count DESC, then file_id ASC -- deterministic, independent of insertion
        # order, so the boundary cut is reproducible across backends.
        ranked = sorted(shared.items(), key=lambda item: (-item[1], item[0]))
        return {file_id for file_id, _count in ranked[:candidate_limit]}

    @staticmethod
    def _resolve_offset_tolerance(
        offset_tolerance: int | None, calibration: Calibration | None
    ) -> int:
        """Resolve the effective banding tolerance (explicit arg wins over calibration).

        ``None`` (the search default) defers to the calibration's
        ``offset_tolerance`` if one was supplied, else ``0`` (OFF). A negative
        tolerance is rejected so the banded window is always well-defined.
        """

        if offset_tolerance is None:
            tolerance = calibration.offset_tolerance if calibration is not None else 0
        else:
            tolerance = int(offset_tolerance)
        if tolerance < 0:
            raise ValueError("offset_tolerance must be non-negative (0 = off/exact)")
        return tolerance

    @staticmethod
    def _banded_winner(
        histogram: dict[int, int] | Counter[int], offset_tolerance: int
    ) -> tuple[int, int]:
        """Return the winning ``(centre_delta, banded_votes)`` for one file.

        With ``offset_tolerance == 0`` this is the exact-bin peak -- the bin with
        the most votes, ties broken by smallest delta -- IDENTICAL to the legacy
        ``max(histogram.items(), key=lambda kv: (kv[1], -kv[0]))``. With
        ``offset_tolerance > 0`` the banded vote count of a candidate centre
        delta ``c`` is the sum of votes over the inclusive window
        ``[c - tol, c + tol]``; the centre maximising that sum wins, ties broken
        by the smaller centre delta. Only deltas actually present in the
        histogram are considered as centres (an empty band can never win), so the
        result stays deterministic and independent of histogram iteration order.
        """

        if offset_tolerance <= 0:
            offset, votes = max(histogram.items(), key=lambda kv: (kv[1], -kv[0]))
            return int(offset), int(votes)
        best_delta = 0
        best_votes = -1
        for centre in histogram:
            banded = sum(
                votes
                for delta, votes in histogram.items()
                if centre - offset_tolerance <= delta <= centre + offset_tolerance
            )
            # votes DESC, then delta ASC -- same deterministic tie-break as the
            # exact-bin path so a degenerate (single-bin) histogram is unchanged.
            if banded > best_votes or (banded == best_votes and centre < best_delta):
                best_votes = banded
                best_delta = centre
        return int(best_delta), int(best_votes)

    @classmethod
    def _reduce_banded_rows(
        cls,
        rows: Iterable[tuple[Any, Any, Any, Any, Any]],
        offset_tolerance: int,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Reduce per-(file, delta, votes) histogram rows to banded aggregates.

        Shared by the SQL backends' banded path: each backend computes the
        compact per-(file, delta) histogram server-side, then this folds it into
        the banded winner via :meth:`_banded_winner` -- the SAME Python code the
        in-memory backend uses -- so the banded ``(offset, aligned_votes)`` is
        cross-backend parity-identical. ``total_votes`` and ``unique_hashes`` are
        carried through unchanged (they are per-file, not per-bin).
        """

        histograms: dict[str, dict[int, int]] = defaultdict(dict)
        totals: dict[str, tuple[int, int]] = {}
        for file_id, delta, votes, total, uniq in rows:
            key = str(file_id)
            histograms[key][int(delta)] = int(votes)
            totals[key] = (int(total), int(uniq))
        aggregates: dict[str, tuple[int, int, int, int]] = {}
        for file_id, histogram in histograms.items():
            offset, aligned = cls._banded_winner(histogram, offset_tolerance)
            total, uniq = totals[file_id]
            aggregates[file_id] = (offset, aligned, total, uniq)
        return aggregates

    def _aggregate(
        self,
        fingerprint: Fingerprint,
        offset_tolerance: int = 0,
        candidates: set[str] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Per-file ``(offset, aligned_votes, total_votes, unique_hashes)``.

        Default in-memory aggregation over a batched :meth:`query_many` fetch.
        The winning offset is chosen by :meth:`_banded_winner` -- with
        ``offset_tolerance == 0`` (the default) the exact-bin peak (most votes,
        ties broken by smallest offset), a deterministic rule SQL backends
        replicate exactly; with ``> 0`` the banded peak over +-tolerance bins.

        ``candidates`` is the OPT-IN prefilter set from :meth:`_select_candidates`:
        ``None`` (the default) aggregates EVERY file sharing a query hash, which
        is byte-identical to before the prefilter existed; a set restricts the
        offset-voting to those ``file_id`` only. Skipping a non-candidate's
        postings cannot change the aggregate of any retained file (votes are
        accumulated per file independently), so a candidate set that is a
        superset of the true top-k yields the identical ranking.
        """

        offset_histograms: dict[str, Counter[int]] = defaultdict(Counter)
        total_votes: Counter[str] = Counter()
        unique_hashes: dict[str, set[int]] = defaultdict(set)

        postings_by_code = self.query_many({qh.hash_code for qh in fingerprint.hashes})
        for query_hash in fingerprint.hashes:
            for posting in postings_by_code.get(query_hash.hash_code, ()):
                if candidates is not None and posting.file_id not in candidates:
                    continue
                offset = posting.time_offset - query_hash.time_offset
                offset_histograms[posting.file_id][offset] += 1
                total_votes[posting.file_id] += 1
                unique_hashes[posting.file_id].add(query_hash.hash_code)

        aggregates: dict[str, tuple[int, int, int, int]] = {}
        for file_id, histogram in offset_histograms.items():
            if not histogram:
                continue
            offset, aligned = self._banded_winner(histogram, offset_tolerance)
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
        """Write a portable JSON snapshot durably (same schema for every backend).

        Crash-safe: the snapshot is written to a temp file in the SAME directory
        (so :func:`os.replace` is atomic on the same filesystem), flushed and
        ``fsync``-ed, then atomically renamed over the destination. After the
        rename the parent directory is ``fsync``-ed (best-effort) so the rename
        itself is durable across power loss. The prior contents (if any) are
        preserved at ``<dest>.bak`` first, so a corrupt or truncated primary can
        fall back on load. A failed write never leaves a partial primary file
        behind. The written JSON carries a ``schema_version`` for forward
        compatibility (see :data:`SNAPSHOT_SCHEMA_VERSION`).
        """

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["schema_version"] = SNAPSHOT_SCHEMA_VERSION
        tmp = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            # Keep a backup of the existing good snapshot before overwriting it.
            if destination.exists():
                backup = destination.with_name(f"{destination.name}.bak")
                shutil.copy2(destination, backup)
            os.replace(tmp, destination)
            # fsync the parent directory so the rename (the directory entry) is
            # durable too. Best-effort: not all platforms/filesystems support
            # opening or fsyncing a directory, so never let this break the save.
            try:
                dir_fd = os.open(str(destination.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _read_snapshot(path: str | Path) -> dict | None:
        """Parse a JSON snapshot, falling back to ``<path>.bak`` if corrupt.

        Returns ``None`` when no primary and no backup exist (a fresh index).
        If the primary is present but unparseable (truncated/corrupt JSON from
        an interrupted write), a valid ``<path>.bak`` is used instead. Raising
        rather than silently returning an empty index here is deliberate: an
        empty index would overwrite the good ``.bak`` on the next save.
        """

        source = Path(path)
        backup = source.with_name(f"{source.name}.bak")
        if not source.exists():
            if not backup.exists():
                return None
            # Primary is gone but a backup exists; a corrupt backup here must not
            # surface as a raw JSONDecodeError -- mirror the corrupt-primary path.
            try:
                with backup.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError) as backup_error:
                raise InvalidSnapshotError(
                    f"index snapshot primary {source} is missing and its backup "
                    f"({backup}) is corrupt"
                ) from backup_error
        try:
            with source.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as primary_error:
            if backup.exists():
                try:
                    with backup.open("r", encoding="utf-8") as handle:
                        return json.load(handle)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            raise InvalidSnapshotError(
                f"index snapshot at {source} is corrupt and no valid backup "
                f"({backup}) was found"
            ) from primary_error

    def load_snapshot(self, path: str | Path) -> HashIndex:
        """Bulk-load a JSON snapshot (from any backend's ``save``) via ``add``."""

        data = self._read_snapshot(path)
        if data is None:
            return self
        if isinstance(data, dict):
            _validate_schema_version(data)
        files = data.get("files", {}) if isinstance(data, dict) else {}
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        if not isinstance(files, dict):
            raise InvalidSnapshotError("invalid index snapshot: files must be a mapping")
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
                and _in_hash_range(int(entry[0]))
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
            for fingerprint in fingerprints:
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
        files = data.get("files", {})
        if not isinstance(files, dict):
            raise InvalidSnapshotError("invalid index: files must be a mapping")
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

        # Collapse to the surviving (last) fingerprint per file_id, preserving
        # the order in which each file_id first appeared for deterministic ops.
        survivors: dict[str, Fingerprint] = {}
        for fingerprint in fingerprints:
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
            pipe.sadd(self._key("files"), file_id)
            pipe.set(self._key("m", file_id), json.dumps(metadata, sort_keys=True))
            for hash_code, time_offset in entries:
                pipe.rpush(self._key("h", str(hash_code)), f"{file_id}:{time_offset}")
                pipe.rpush(self._key("f", file_id), f"{hash_code}:{time_offset}")
            net_delta += len(entries)

        if net_delta:
            pipe.incrby(self._key("npostings"), net_delta)
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


class SQLiteHashIndex(HashIndex):
    """SQLite-backed hash index: zero-dependency (stdlib), file-persistent.

    Implements the same :class:`HashIndex` contract and inherits the shared
    ``search``/``save``/``load_snapshot``. Postings live in a relational table
    indexed on ``hash_code`` (fast ``query``) and ``file_id`` (fast ``remove``).
    Pass ``":memory:"`` for an ephemeral in-process database (used by tests) or a
    file path for persistence; or inject an existing ``sqlite3.Connection``.

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

    def add_many(self, fingerprints: Iterable[Fingerprint]) -> None:
        """Ingest the whole batch in ONE transaction (one commit, not per file).

        Equivalent to per-item :meth:`add`: each ``file_id`` is removed before it
        is (re)inserted, so a duplicate ``file_id`` within the batch keeps only
        the LAST fingerprint and any pre-batch rows for it are deleted exactly
        once -- identical resulting postings, metadata, and search results. The
        single commit (instead of one fsync-bound commit per file) plus a single
        ``executemany`` for all postings is the bulk-ingest win.
        """

        # Last-wins per file_id: a sequential add() of a repeated file_id removes
        # the earlier copy, so only the final fingerprint survives. Collapsing
        # here also avoids the files PRIMARY KEY conflict a naive re-insert hits.
        survivors: dict[str, Fingerprint] = {}
        for fingerprint in fingerprints:
            survivors[fingerprint.file_id] = fingerprint
        if not survivors:
            return

        file_rows: list[tuple[str, str]] = []
        posting_rows: list[tuple[str, int, int]] = []
        for file_id, fingerprint in survivors.items():
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
            file_rows.append((file_id, json.dumps(metadata, sort_keys=True)))
            posting_rows.extend(
                (file_id, self._encode(item.hash_code), int(item.time_offset))
                for item in fingerprint.hashes
            )

        try:
            # remove() effect for every target file_id (pre-batch rows), batched.
            self._conn.executemany(
                "DELETE FROM postings WHERE file_id = ?", [(fid,) for fid in survivors]
            )
            self._conn.executemany(
                "DELETE FROM files WHERE file_id = ?", [(fid,) for fid in survivors]
            )
            self._conn.executemany(
                "INSERT INTO files (file_id, metadata) VALUES (?, ?)", file_rows
            )
            self._conn.executemany(
                "INSERT INTO postings (file_id, hash_code, time_offset) VALUES (?, ?, ?)",
                posting_rows,
            )
        except BaseException:
            # Keep the all-or-nothing contract: a mid-batch failure rolls the
            # whole transaction back instead of leaving a partial ingest.
            self._conn.rollback()
            raise
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

    def _aggregate(
        self,
        fingerprint: Fingerprint,
        offset_tolerance: int = 0,
        candidates: set[str] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Aggregate the offset histogram server-side via a single SQL pass.

        Loads the query's (hash_code, offset) pairs into a temp table, joins to
        the postings, and groups by (file_id, delta) -- so only per-file
        aggregates cross the boundary, not millions of postings.

        With ``offset_tolerance == 0`` (the default) the winning bin is picked
        server-side as votes DESC then delta ASC, matching the base in-memory
        tie-break exactly -- this path is unchanged and BYTE-IDENTICAL to before
        the option existed. With ``> 0`` the SQL still groups to per-(file, delta)
        bins server-side (only the compact histogram crosses the boundary, never
        raw postings), but the banded winner is chosen in Python via the shared
        :meth:`_banded_winner`, so every backend bands identically.

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
                        SELECT p.file_id AS file_id, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                        FROM postings p JOIN _query q ON p.hash_code = q.hash_code
                    ),
                    bins AS (SELECT file_id, delta, COUNT(*) AS votes FROM matches GROUP BY file_id, delta),
                    totals AS (
                        SELECT file_id, COUNT(*) AS total_votes, COUNT(DISTINCT hc) AS uniq
                        FROM matches GROUP BY file_id
                    )
                    SELECT b.file_id, b.delta, b.votes, t.total_votes, t.uniq
                    FROM bins b JOIN totals t ON b.file_id = t.file_id
                    """
                ).fetchall()
            else:
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

    def list_files(self) -> list[str]:
        # ORDER BY in SQL gives the same ascending order as the base sorted()
        # contract without a Python-side sort of the whole id set.
        return [
            row[0]
            for row in self._conn.execute("SELECT file_id FROM files ORDER BY file_id").fetchall()
        ]

    def contains(self, file_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM files WHERE file_id = ? LIMIT 1", (file_id,)
        ).fetchone()
        return row is not None

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

    def __enter__(self) -> SQLiteHashIndex:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


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
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._files_table}")
                count = int(cur.fetchone()[0])
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return count

    @property
    def posting_count(self) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._postings_table}")
                count = int(cur.fetchone()[0])
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return count

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

    def add_many(self, fingerprints: Iterable[Fingerprint]) -> None:
        """Ingest the whole batch in ONE transaction, streaming postings via COPY.

        Equivalent to per-item :meth:`add`: each ``file_id`` is removed before it
        is (re)inserted, so a duplicate ``file_id`` keeps only the LAST
        fingerprint and any pre-batch rows for it are deleted once -- identical
        postings, metadata, and search results. Postings stream through
        ``cursor.copy`` (PostgreSQL's bulk path, far faster than row-at-a-time
        INSERT); metadata goes via ``executemany``; everything commits once.
        """

        # Last-wins per file_id (a sequential add() of a repeated id removes the
        # earlier copy), which also avoids the files PRIMARY KEY conflict.
        survivors: dict[str, Fingerprint] = {}
        for fingerprint in fingerprints:
            survivors[fingerprint.file_id] = fingerprint
        if not survivors:
            return

        file_rows: list[tuple[str, str]] = []
        posting_rows: list[tuple[str, int, int]] = []
        for file_id, fingerprint in survivors.items():
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
            file_rows.append((file_id, json.dumps(metadata, sort_keys=True)))
            posting_rows.extend(
                (file_id, self._encode(item.hash_code), int(item.time_offset))
                for item in fingerprint.hashes
            )

        file_ids = list(survivors)
        try:
            with self._conn.cursor() as cur:
                # remove() effect for every target file_id (pre-batch rows).
                cur.execute(
                    f"DELETE FROM {self._postings_table} WHERE file_id = ANY(%s)", (file_ids,)
                )
                cur.execute(
                    f"DELETE FROM {self._files_table} WHERE file_id = ANY(%s)", (file_ids,)
                )
                cur.executemany(
                    f"INSERT INTO {self._files_table} (file_id, metadata) VALUES (%s, %s::jsonb)",
                    file_rows,
                )
                if posting_rows:
                    with cur.copy(
                        f"COPY {self._postings_table} (file_id, hash_code, time_offset) FROM STDIN"
                    ) as copy:
                        for row in posting_rows:
                            copy.write_row(row)
        except BaseException:
            self._conn.rollback()  # all-or-nothing; never leave a partial ingest
            raise
        self._conn.commit()

    def remove(self, file_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._postings_table} WHERE file_id = %s", (file_id,))
            cur.execute(f"DELETE FROM {self._files_table} WHERE file_id = %s", (file_id,))
        self._conn.commit()

    def query(self, hash_code: int) -> list[IndexPosting]:
        code = int(hash_code)
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT file_id, time_offset FROM {self._postings_table} WHERE hash_code = %s",
                    (self._encode(code),),
                )
                rows = cur.fetchall()
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
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
        try:
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
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return results

    def _aggregate(
        self,
        fingerprint: Fingerprint,
        offset_tolerance: int = 0,
        candidates: set[str] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Aggregate the offset histogram server-side in one round-trip.

        The query's (hash_code, offset) pairs are passed as two arrays and
        unnested into a derived table, joined to the postings, then grouped by
        (file_id, delta). Only per-file aggregates return.

        With ``offset_tolerance == 0`` (the default) the winning bin is picked
        server-side as votes DESC then delta ASC, matching the base in-memory
        tie-break -- unchanged and BYTE-IDENTICAL to before the option existed.
        With ``> 0`` the compact per-(file, delta) histogram is returned and the
        banded winner is chosen by the shared :meth:`_banded_winner` so every
        backend bands identically.

        ``candidates`` is the OPT-IN prefilter set: ``None`` (the default) returns
        the aggregate for every matched file, byte-identical to before the
        prefilter existed; a set keeps only those ``file_id`` (filtered on the
        per-file aggregate rows, so a retained file's row is never altered), so a
        candidate superset of the true top-k yields the identical ranking.
        """

        if not fingerprint.hashes:
            return {}
        codes = [self._encode(h.hash_code) for h in fingerprint.hashes]
        offsets = [int(h.time_offset) for h in fingerprint.hashes]
        banded = offset_tolerance > 0
        if banded:
            sql = f"""
                WITH q(hash_code, qoff) AS (SELECT * FROM unnest(%s::bigint[], %s::int[])),
                matches AS (
                    SELECT p.file_id AS file_id, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                    FROM {self._postings_table} p JOIN q ON p.hash_code = q.hash_code
                ),
                bins AS (SELECT file_id, delta, COUNT(*) AS votes FROM matches GROUP BY file_id, delta),
                totals AS (
                    SELECT file_id, COUNT(*) AS total_votes, COUNT(DISTINCT hc) AS uniq
                    FROM matches GROUP BY file_id
                )
                SELECT b.file_id, b.delta, b.votes, t.total_votes, t.uniq
                FROM bins b JOIN totals t ON b.file_id = t.file_id
                """
        else:
            sql = f"""
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
                """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (codes, offsets))
                rows = cur.fetchall()
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        if candidates is not None:
            rows = [row for row in rows if str(row[0]) in candidates]
        if banded:
            return self._reduce_banded_rows(rows, offset_tolerance)
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
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT metadata FROM {self._files_table} WHERE file_id = %s", (file_id,)
                )
                row = cur.fetchone()
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        if not row:
            return {}
        data = row[0]
        if isinstance(data, str):  # some drivers return JSONB as text
            try:
                data = json.loads(data)
            except (ValueError, TypeError):
                return {}
        return data if isinstance(data, dict) else {}

    def list_files(self) -> list[str]:
        try:
            with self._conn.cursor() as cur:
                # ORDER BY file_id matches the base sorted() contract server-side.
                cur.execute(f"SELECT file_id FROM {self._files_table} ORDER BY file_id")
                file_ids = [row[0] for row in cur.fetchall()]
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return file_ids

    def contains(self, file_id: str) -> bool:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT 1 FROM {self._files_table} WHERE file_id = %s LIMIT 1", (file_id,)
                )
                row = cur.fetchone()
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return row is not None

    def to_dict(self) -> dict[str, object]:
        files: dict[str, list[list[int]]] = {}
        metadata: dict[str, dict] = {}
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT file_id FROM {self._files_table}")
                file_ids = [row[0] for row in cur.fetchall()]
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
        finally:
            # Even an EMPTY index (only the file_id SELECT runs, no per-file
            # reads that would roll back) must not leave the connection
            # idle-in-transaction holding a snapshot.
            self._conn.rollback()
        return {"backend": "postgres", "files": files, "metadata": metadata}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PostgresHashIndex:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
