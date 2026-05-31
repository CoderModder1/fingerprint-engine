"""Storage-agnostic HashIndex base: search/scoring, persistence, format-version."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..exceptions import (
    FormatVersionMismatchError,
    InvalidSnapshotError,
    SnapshotWriteRefused,
)
from ..models import (
    FINGERPRINT_FORMAT_VERSION,
    FORMAT_VERSION_KEY,
    Calibration,
    ConstellationHash,
    Fingerprint,
    IndexPosting,
    SearchResult,
)
from ._common import (
    SNAPSHOT_FORMAT_VERSION_KEY,
    SNAPSHOT_SCHEMA_VERSION,
    _fsync_path,
    _in_hash_range,
    _snapshot_format_version,
    _validate_schema_version,
)

logger = logging.getLogger(__name__)


class HashIndex(ABC):
    """Storage-agnostic contract for searchable fingerprint postings."""

    # Hash-derivation format version of this index's postings, and whether it
    # has been pinned by a recorded fingerprint/snapshot yet. Both default at
    # the class level (lazily, so a backend that never sets them -- e.g. an
    # empty or legacy index -- reads the engine baseline and is unpinned), and
    # are plain instance attributes so they cost nothing per posting and are
    # independent of the storage backend.
    _format_version: int = FINGERPRINT_FORMAT_VERSION
    _format_version_pinned: bool = False
    # Cross-format ADD policy (F1). ``False`` (the default) FAILS CLOSED: adding a
    # fingerprint whose hash format version differs from this index's pinned
    # version raises :class:`FormatVersionMismatchError`, so postings from an
    # incompatible code space can never silently contaminate the index. Set
    # ``True`` to restore the legacy fail-soft behaviour (a :class:`RuntimeWarning`,
    # first-writer-wins). A default single-config corpus reports one version, so
    # neither branch is ever reached for it and the add path stays byte-identical.
    allow_format_mixing: bool = False

    @property
    def format_version(self) -> int:
        """The hash-derivation format version of the postings in this index.

        Pinned from the :attr:`Fingerprint.format_version` of the FIRST
        fingerprint added (and restored from a loaded snapshot's
        :data:`SNAPSHOT_FORMAT_VERSION_KEY`). A query must share this version for
        its hash codes to mean the same thing as the index's;
        :meth:`search` surfaces a mismatch. An empty/legacy index reports the
        default :data:`FINGERPRINT_FORMAT_VERSION`.
        """

        return self._format_version

    def _record_format_version(self, fingerprint: Fingerprint) -> None:
        """Pin (or check against) the index's format version from a fingerprint.

        The first recorded fingerprint pins the index's format version, and -- for
        a NON-default version on a durable backend -- persists it via
        :meth:`_persist_format_version` so reopening the store restores the corpus
        version instead of silently defaulting to the engine baseline (F2). A later
        fingerprint whose recorded version DIFFERS is an attempt to mix
        incompatible hash derivations in one index: their codes do not share a
        space, so any later "match" between them is invalid, not merely weak. By
        default this FAILS CLOSED (F1) -- it raises
        :class:`FormatVersionMismatchError` so the contaminating postings never
        enter the index. Setting :attr:`allow_format_mixing` restores the legacy
        fail-soft behaviour (a :class:`RuntimeWarning`, first-writer-wins). A
        default single-config corpus reports one version, so neither the persist
        nor the mismatch branch is ever reached and the add path is byte-identical.
        """

        incoming = fingerprint.format_version
        if not self._format_version_pinned:
            self._format_version = incoming
            self._format_version_pinned = True
            # The default version needs no durable record: an absent meta is read
            # back AS the default (see _load_persisted_format_version), so only a
            # non-default corpus writes the meta -- keeping the default path's
            # storage and commit pattern byte-identical.
            if incoming != FINGERPRINT_FORMAT_VERSION:
                self._persist_format_version(incoming)
            return
        if incoming != self._format_version:
            message = self._format_mismatch_message(incoming, self._format_version)
            if self.allow_format_mixing:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
                return
            raise FormatVersionMismatchError(
                message, query_version=incoming, index_version=self._format_version
            )

    @staticmethod
    def _format_mismatch_message(incoming: int, index_version: int) -> str:
        return (
            f"adding a fingerprint with hash format version {incoming} to an "
            f"index built at version {index_version}; their hash codes "
            "do not share a code space, so matches between them are invalid. "
            "Re-index with a single, consistent FingerprintConfig."
        )

    def _validate_batch_format(self, fingerprints: list[Fingerprint]) -> None:
        """Fail-closed PRE-CHECK for :meth:`add_many`: verify the whole batch.

        Confirms every member shares ONE format version and matches the index's
        pinned version (the first member sets the tentative version for an as-yet
        unpinned index), WITHOUT pinning or persisting anything. Calling it before
        any mutation makes ``add_many`` truly all-or-nothing on a cross-format
        batch: a rejected batch leaves the index -- and its DURABLE version stamp
        -- completely untouched, instead of pinning/persisting the first member's
        version off a batch that then rolls back. A no-op when
        :attr:`allow_format_mixing` is set (that legacy path warns per member
        during the write and proceeds, first-writer-wins).
        """

        if self.allow_format_mixing:
            return
        reference = self._format_version if self._format_version_pinned else None
        for fingerprint in fingerprints:
            incoming = fingerprint.format_version
            if reference is None:
                reference = incoming
                continue
            if incoming != reference:
                raise FormatVersionMismatchError(
                    self._format_mismatch_message(incoming, reference),
                    query_version=incoming,
                    index_version=reference,
                )

    def _persist_format_version(self, version: int) -> None:  # noqa: B027 - intentional no-op hook
        """Hook: durably record the just-pinned NON-default format version.

        Default no-op: the in-memory backend has no durable store of its own (its
        version travels with the JSON snapshot via :meth:`save` /
        :meth:`load_snapshot`). Durable backends (SQLite/Redis/Postgres) override
        this to write the version into a side meta table/key so a reopen restores
        the corpus version. Only ever called for a non-default version, so it never
        touches storage on the default path.
        """

    def _load_persisted_format_version(self, raw: object) -> None:
        """Pin the format version a durable backend read back when it opened.

        Mirrors :func:`_snapshot_format_version`'s tolerant rule: a parseable int
        pins that version; an absent/garbage value leaves the index at the unpinned
        engine default, so a legacy store written before the meta existed behaves
        exactly as today until its next write re-pins (and persists) the version.
        """

        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            return
        try:
            version = int(raw)
        except ValueError:
            return
        self._format_version = version
        self._format_version_pinned = True

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
        strict_format: bool = False,
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
        prefilter: files are first ranked by SHARED-POSTING COUNT (for each file,
        the sum over shared codes of query-multiplicity x file-posting-count -- a
        single batched posting lookup, no per-posting offset arithmetic), and only
        the top ``candidate_limit`` files proceed to the exact offset-histogram
        aggregation and scoring.

        Exactness/recall trade-off: that shared-posting count is the file's TOTAL
        offset-histogram votes across all offsets, hence a monotone UPPER BOUND on
        its ALIGNED votes (the count at the single best offset), so for a
        high-overlap match -- a self-match, a near-duplicate, even one dominated
        by a single code repeated at many coherent offsets -- the true file is
        guaranteed to be among the highest
        shared-posting-count files and thus in the candidate set. When the
        candidate set is a SUPERSET of the true top-``top_k`` (the normal case for
        a generous ``candidate_limit``), the final ranking is IDENTICAL to full
        search: scoring runs unchanged on the surviving files and the dropped
        files could never have out-ranked them. The only files a too-tight limit
        can drop are *low*-overlap ones -- weak partial matches that share few
        postings but happen to align coherently -- so set ``candidate_limit``
        comfortably above ``top_k`` to keep recall at 1.0 for genuine matches.
        Ties in shared-posting count at the cut boundary are broken
        deterministically (count DESC, then file_id ASC) so the set is reproducible.

        ``strict_format`` controls the HASH-FORMAT compatibility check. The query
        fingerprint's :attr:`Fingerprint.format_version` is compared with this
        index's :attr:`format_version`; a mismatch means the two were derived
        under different hash rules, so their codes do not share a code space and
        any "match" is a false result. ``False`` (the default) only emits a
        :class:`RuntimeWarning` and proceeds -- so the new check never breaks an
        existing pipeline and the returned rankings for a MATCHING-format query
        are BYTE-IDENTICAL to before this parameter existed (no warning, no
        behavior change). ``True`` raises :class:`FormatVersionMismatchError`
        instead, for callers that want a hard guard against cross-format search.
        """

        self._check_format_version(fingerprint, strict_format)
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

    def _check_format_version(self, fingerprint: Fingerprint, strict: bool) -> None:
        """Detect a cross-format query against this index; warn or raise.

        No-op when the query's format version matches the index's (the normal
        case -- so a same-format search is byte-identical to before this check),
        or when the index is empty/unpinned (nothing to be incompatible with).
        On a mismatch: raise :class:`FormatVersionMismatchError` if ``strict``,
        else emit a :class:`RuntimeWarning`. Matching across formats is invalid
        because the hash codes are derived under different rules and do not share
        a code space.
        """

        query_version = fingerprint.format_version
        index_version = self._format_version
        if query_version == index_version:
            return
        message = (
            f"query fingerprint hash format version {query_version} does not match "
            f"this index's version {index_version}; their hash codes are derived "
            "under different rules and do not share a code space, so any match "
            "between them is invalid. Re-index or re-fingerprint with a single, "
            "consistent FingerprintConfig."
        )
        if strict:
            raise FormatVersionMismatchError(
                message, query_version=query_version, index_version=index_version
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _select_candidates(
        self, fingerprint: Fingerprint, candidate_limit: int | None
    ) -> set[str] | None:
        """Cheap shared-posting candidate prefilter; ``None`` (default) = OFF.

        Returns ``None`` when ``candidate_limit`` is ``None`` (the default) so the
        caller runs the full exact search on every file -- the byte-identical
        legacy path. When ``candidate_limit`` is a non-negative int, returns the
        set of the top-``candidate_limit`` ``file_id`` ranked by SHARED-POSTING
        COUNT: for each file, ``sum over shared codes of (query multiplicity x
        file posting count)``.

        That sum is the file's TOTAL offset-histogram votes across all offsets,
        so it is a genuine monotone UPPER BOUND on the file's ALIGNED votes
        (aligned votes are the count at the single best offset, which can never
        exceed the total). A high-overlap match -- including one dominated by a
        single code repeated at many coherent offsets -- is therefore always
        retained, so a generous ``candidate_limit`` keeps recall at 1.0. (The
        earlier prefilter counted only DISTINCT shared codes, +1 per code; that
        UNDER-counts a repeated-code match -- a code at N coherent offsets is N
        aligned votes but only +1 -- so a too-tight limit could silently drop the
        true #1. Counting postings, not distinct codes, fixes that.)

        Computed from the SAME batched :meth:`query_many` fetch the exact
        aggregation would do, but without the per-posting offset arithmetic, so
        it stays the cheap prefilter. The cut is deterministic: shared-posting
        count DESC, then file_id ASC, so a boundary tie is reproducible across
        backends.

        ``candidate_limit == 0`` selects no files (an explicit empty search). A
        negative limit is rejected so the cut is always well defined.
        """

        if candidate_limit is None:
            return None
        if candidate_limit < 0:
            raise ValueError("candidate_limit must be non-negative or None (None = off)")
        # Query-side multiplicity per code (duplicates matter: a code occurring
        # k times in the query contributes k votes per matching file posting).
        query_mult: Counter[int] = Counter(qh.hash_code for qh in fingerprint.hashes)
        shared: Counter[str] = Counter()
        postings_by_code = self.query_many(set(query_mult))
        for code, postings in postings_by_code.items():
            qm = query_mult[code]
            # File-side multiplicity of this code; qm * fm is the code's total
            # vote contribution (across all offsets) for that file. Summed over
            # codes this dominates aligned_votes -- a true upper bound.
            for file_id, fm in Counter(posting.file_id for posting in postings).items():
                shared[file_id] += qm * fm
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

    def save(self, path: str | Path, *, force: bool = False) -> None:
        """Write a portable JSON snapshot durably (same schema for every backend).

        Crash-safe: the snapshot is written to a temp file in the SAME directory
        (so :func:`os.replace` is atomic on the same filesystem), flushed and
        ``fsync``-ed, then atomically renamed over the destination. After the
        rename the parent directory is ``fsync``-ed (best-effort) so the rename
        itself is durable across power loss. The prior contents (if any) are
        copied to ``<dest>.bak`` first and that backup's data is ``fsync``-ed
        before the primary is replaced, so a corrupt or truncated primary can
        fall back on load even across a crash mid-replace. A failed write never
        leaves a partial primary file behind. The written JSON carries a
        ``schema_version`` for forward compatibility (see
        :data:`SNAPSHOT_SCHEMA_VERSION`).

        Data-loss guard: an EMPTY (zero-file) snapshot is refused with
        :class:`SnapshotWriteRefused` when it would overwrite an existing
        NON-EMPTY primary -- a freshly created, emptied, or failed-rebuild index
        saved over a populated corpus would otherwise clobber the only good copy
        (the empty primary loads cleanly, so the ``.bak`` fallback never fires).
        Pass ``force=True`` to override when emptying the snapshot is intended.

        Concurrency: the temp file is uniquely named per writer (pid + random),
        so two threads saving the same destination cannot tear or unlink each
        other's in-flight write; the last full save wins atomically.
        """

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["schema_version"] = SNAPSHOT_SCHEMA_VERSION
        # Travel the hash-derivation format version with the snapshot so a load
        # restores the index's version and a later search can detect a
        # cross-format query. Distinct from schema_version (see the module
        # constants). Additive top-level field: it does not touch postings,
        # metadata, or the schema container, so the round-tripped index is
        # otherwise byte-identical.
        payload[SNAPSHOT_FORMAT_VERSION_KEY] = self._format_version

        # Refuse a destructive empty save (see the data-loss guard above). Only
        # an empty-over-non-empty save is blocked, so a normal save -- and an
        # empty save over an absent/empty/corrupt primary -- is unaffected.
        if not force and not payload.get("files"):
            existing_count = self._existing_snapshot_file_count(destination)
            if existing_count > 0:
                raise SnapshotWriteRefused(
                    f"refusing to overwrite a non-empty index snapshot at "
                    f"{destination} ({existing_count} files) with an EMPTY one; "
                    "pass force=True if emptying the snapshot is intentional",
                    existing_file_count=existing_count,
                )

        tmp = destination.with_name(f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            # Keep a backup of the existing good snapshot before overwriting it,
            # and fsync the backup's DATA so the crash-safe fallback is durable
            # before the primary is swapped (the dir fsync below covers its
            # directory entry).
            if destination.exists():
                backup = destination.with_name(f"{destination.name}.bak")
                shutil.copy2(destination, backup)
                _fsync_path(backup)
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
    def _existing_snapshot_file_count(destination: Path) -> int:
        """Best-effort file count of the existing primary snapshot (0 if none/unreadable).

        Used only by the empty-save guard in :meth:`save`. A missing or
        unparseable primary returns 0 so the guard never blocks overwriting a
        corrupt or absent file -- it protects only a readable, non-empty corpus.
        """

        if not destination.exists():
            return 0
        try:
            with destination.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return 0
        if not isinstance(existing, dict):
            return 0
        files = existing.get("files")
        return len(files) if isinstance(files, dict) else 0

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
        # Restore the index's hash-derivation format version from the snapshot
        # (default for a legacy/absent field) so a later search detects a
        # cross-format query. Pin it directly rather than relying on a rebuilt
        # fingerprint's config, so even an EMPTY snapshot carries its version.
        snapshot_version = (
            _snapshot_format_version(data) if isinstance(data, dict) else FINGERPRINT_FORMAT_VERSION
        )
        self._format_version = snapshot_version
        self._format_version_pinned = True
        files = data.get("files", {}) if isinstance(data, dict) else {}
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        if not isinstance(files, dict):
            raise InvalidSnapshotError("invalid index snapshot: files must be a mapping")
        total_dropped = 0
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
            # Surface (don't silence) malformed/out-of-range postings: a partial
            # load is still useful, but a SILENT drop hid a degraded snapshot that
            # then returned weaker results. Rebuilding via add() below recomputes
            # hash_count from the kept hashes, so confidence stays calibrated here.
            dropped = len(entries) - len(hashes)
            if dropped:
                total_dropped += dropped
                logger.warning(
                    "snapshot file %s: skipped %d malformed/out-of-range posting(s) on load",
                    file_id,
                    dropped,
                )
            meta = metadata.get(file_id, {}) if isinstance(metadata, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            self.add(
                Fingerprint(
                    file_id=str(file_id),
                    path=str(meta.get("path", "")),
                    handler=str(meta.get("handler", "")),
                    size_bytes=int(meta.get("size_bytes", 0) or 0),
                    content_sha256=str(meta.get("content_sha256", file_id)),
                    # Stamp the snapshot's format version so the rebuilt
                    # fingerprint reports it (matching the index we just pinned),
                    # and a re-add never spuriously warns about a mismatch.
                    config={FORMAT_VERSION_KEY: snapshot_version},
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
        if total_dropped:
            logger.warning(
                "loaded snapshot %s with %d posting(s) skipped as malformed or "
                "out-of-range; the index is degraded -- re-fingerprint the source "
                "to restore full recall",
                path,
                total_dropped,
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
