"""Hash index interfaces and the default in-memory backend."""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar, cast

from .exceptions import (
    FormatVersionMismatchError,
    InvalidSnapshotError,
    SnapshotWriteRefused,
)
from .models import (
    FINGERPRINT_FORMAT_VERSION,
    FORMAT_VERSION_KEY,
    Calibration,
    ConstellationHash,
    Fingerprint,
    IndexPosting,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Schema version stamped into every snapshot written by :meth:`HashIndex.save`.
# A snapshot whose top-level ``schema_version`` is present but not in
# ``_SUPPORTED_SCHEMA_VERSIONS`` is rejected on load; an ABSENT version is
# treated as version 1 for backward compatibility with already-written files.
SNAPSHOT_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

# Top-level snapshot field recording the HASH-DERIVATION format version of the
# indexed postings (see ``FINGERPRINT_FORMAT_VERSION`` in ``core/models.py``).
# DISTINCT from ``schema_version``: schema versions the JSON container, this
# versions the meaning of the ``hash_code`` integers inside it. An ABSENT field
# is treated as the default ``FINGERPRINT_FORMAT_VERSION`` (legacy snapshots
# written before the field existed stay loadable and compatible).
SNAPSHOT_FORMAT_VERSION_KEY = "fingerprint_format_version"

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


def _snapshot_format_version(data: dict[str, object]) -> int:
    """Read a snapshot's hash-derivation format version.

    An ABSENT :data:`SNAPSHOT_FORMAT_VERSION_KEY` is treated as the default
    :data:`FINGERPRINT_FORMAT_VERSION` -- legacy snapshots written before the
    field existed describe the default derivation, so they stay loadable and
    compatible. A malformed (non-int) value also falls back to the default
    rather than aborting the load; the version is advisory metadata for the
    search-time compatibility check, never a gate on loadability.
    """

    raw = data.get(SNAPSHOT_FORMAT_VERSION_KEY, FINGERPRINT_FORMAT_VERSION)
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return FINGERPRINT_FORMAT_VERSION
    try:
        return int(raw)
    except ValueError:
        return FINGERPRINT_FORMAT_VERSION


def _in_hash_range(hash_code: int) -> bool:
    """Whether ``hash_code`` fits the unsigned 64-bit posting range."""

    return _HASH_CODE_MIN <= hash_code <= _HASH_CODE_MAX


def _fsync_path(path: Path) -> None:
    """Best-effort fsync of a file's data blocks (never raises on an unsupported FS).

    Used to make a freshly written backup durable before the primary is
    replaced. Like the parent-directory fsync, this is best-effort: not every
    platform/filesystem supports it, so a failure is swallowed rather than
    allowed to break the save.
    """

    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _index_metadata(fingerprint: Fingerprint) -> dict[str, Any]:
    """The per-file metadata dict every backend stores for a fingerprint.

    Single source of truth for the index-metadata shape, so all four backends
    store an identical record -- which is what makes list_files/iter_metadata and
    snapshot interop parity hold. ``**fingerprint.metadata`` is merged LAST so a
    handler's extra keys never shadow the canonical fields. The SQL/Redis backends
    json.dumps(..., sort_keys=True) it (so insertion order is normalized there);
    InMemory stores it directly and to_dict serializes it.
    """

    return {
        "file_id": fingerprint.file_id,
        "path": fingerprint.path,
        "handler": fingerprint.handler,
        "size_bytes": fingerprint.size_bytes,
        "content_sha256": fingerprint.content_sha256,
        "hash_count": fingerprint.hash_count,
        "landmark_count": fingerprint.landmark_count,
        **fingerprint.metadata,
    }


_BackendMethod = TypeVar("_BackendMethod", bound=Callable[..., Any])


def _synchronized(method: _BackendMethod) -> _BackendMethod:
    """Run a backend method while holding ``self._lock`` (the per-index RLock).

    Used by the SQL backends (F3): a single shared DB connection is one serial
    command stream, so concurrent FastAPI-threadpool requests would otherwise
    interleave the multi-statement critical sections -- the SQLite ``_query`` temp
    table, the write transactions -- and corrupt results or raise ``InterfaceError``.
    Serializing every connection-touching method closes that race. The lock is
    re-entrant (``RLock``), so a method that calls another synchronized method
    (``add`` -> ``remove``, ``to_dict`` -> ``_metadata_for``) never self-deadlocks,
    and it is uncontended on the single-threaded path so output is byte-identical.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast(_BackendMethod, wrapper)


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


class PostgresHashIndex(HashIndex):
    """PostgreSQL-backed hash index for a shared, durable, server-grade store.

    Implements the same :class:`HashIndex` contract and inherits the shared
    ``search``/``save``/``load_snapshot``. Pass a ``dsn`` (libpq connection
    string) or inject a ``psycopg`` connection. ``psycopg`` is imported lazily, so
    importing this module never requires it.

    Storage layout (internal, output-preserving): mirroring SQLite, the
    ``files`` table maps each ``file_id`` to a small ``BIGINT`` surrogate id
    (``GENERATED BY DEFAULT AS IDENTITY``), and each posting stores that integer
    ``file_ref`` foreign key rather than the 64-char ``file_id`` string. Every
    read JOINs ``postings`` to ``files`` to recover the original ``file_id``, so
    :meth:`search`, :meth:`to_dict`, :meth:`list_files`, :meth:`query`,
    :meth:`query_many`, the counts, and cross-backend parity are BYTE-IDENTICAL to
    storing the string verbatim. ``postings`` is indexed on ``hash_code`` (fast
    ``query``) and ``file_ref`` (fast ``remove``/join). Metadata is ``JSONB``.

    Migration: a table set written by a PRIOR version of this class has the OLD
    schema (``postings.file_id TEXT``, ``files`` with no ``id`` column).
    :meth:`_init_schema` detects that and migrates it in place, transactionally,
    once on open (see :meth:`_migrate_legacy_schema_if_present`). This backend
    cannot be exercised in CI without a live server (it is gated behind
    ``@requires_pg``), so the migration here is implemented structurally to mirror
    the SQLite path that IS tested live.

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
        self._meta_table = f"{table_prefix}_meta"
        # Serializes the single shared psycopg connection (F3). A psycopg
        # connection is not safe for concurrent use across threads, so the FastAPI
        # threadpool sharing one active_index would otherwise race; mirroring the
        # SQLite backend, every connection-touching method holds this re-entrant
        # lock. Uncontended single-threaded, so output is byte-identical.
        self._lock = threading.RLock()
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
        # Migrate an OLD-schema table set (postings.file_id TEXT, files without a
        # surrogate id) in place, once, before idempotently ensuring the new
        # tables/indexes below.
        self._migrate_legacy_schema_if_present()
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._files_table} ("
                "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, "
                "file_id TEXT UNIQUE NOT NULL, metadata JSONB NOT NULL)"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._postings_table} ("
                "file_ref BIGINT NOT NULL, hash_code BIGINT NOT NULL, time_offset INTEGER NOT NULL)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._postings_table}_hash_idx "
                f"ON {self._postings_table}(hash_code)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._postings_table}_file_idx "
                f"ON {self._postings_table}(file_ref)"
            )
            # Side table recording the corpus hash-format version (F2): a row is
            # written only for a NON-default corpus, so a default store keeps an
            # empty meta and is byte-identical. Never read by to_dict/search.
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._meta_table} "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            cur.execute(
                f"SELECT value FROM {self._meta_table} WHERE key = %s",
                (SNAPSHOT_FORMAT_VERSION_KEY,),
            )
            row = cur.fetchone()
        self._conn.commit()
        # Restore the durably-stored version (F2); absent -> unpinned default.
        if row is not None:
            self._load_persisted_format_version(row[0])

    def _persist_format_version(self, version: int) -> None:
        # F2: upsert the pinned (non-default) version into the side meta table so
        # a reopen restores it. Self-contained commit (only on first pin of a
        # non-default corpus) so it never leaves a dangling transaction.
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self._meta_table} (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (SNAPSHOT_FORMAT_VERSION_KEY, str(version)),
                )
            self._conn.commit()

    def _migrate_legacy_schema_if_present(self) -> None:
        """One-time, transactional, in-place upgrade of an OLD-schema table set.

        Detects the pre-surrogate layout -- a ``postings`` table that HAS a
        ``file_id`` column and LACKS ``file_ref`` -- and rewrites it: a new
        ``files`` table with a ``BIGINT GENERATED ... AS IDENTITY`` surrogate id,
        populated from the distinct old file_ids; a new ``postings`` table whose
        ``file_ref`` is each posting's resolved surrogate; then drop the old
        tables and rename the new ones into place. Every file_id, metadata blob,
        and posting survives; the whole rewrite is one transaction, so a failure
        rolls back to the original old-schema tables. A fresh or already-migrated
        table set (no postings table, or one that already has ``file_ref``) is a
        no-op. Mirrors :meth:`SQLiteHashIndex._migrate_legacy_schema_if_present`;
        gated behind ``@requires_pg`` so it is not run live in CI.
        """

        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (self._postings_table,),
            )
            columns = {row[0] for row in cur.fetchall()}
        if not columns or "file_ref" in columns or "file_id" not in columns:
            self._conn.rollback()  # release the read txn; nothing to migrate
            return
        files_new = f"{self._files_table}_new"
        postings_new = f"{self._postings_table}_new"
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"CREATE TABLE {files_new} ("
                    "id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, "
                    "file_id TEXT UNIQUE NOT NULL, metadata JSONB NOT NULL)"
                )
                cur.execute(
                    f"CREATE TABLE {postings_new} ("
                    "file_ref BIGINT NOT NULL, hash_code BIGINT NOT NULL, time_offset INTEGER NOT NULL)"
                )
                # Identity ids are assigned as the distinct file_ids are inserted.
                cur.execute(
                    f"INSERT INTO {files_new} (file_id, metadata) "
                    f"SELECT file_id, metadata FROM {self._files_table}"
                )
                cur.execute(
                    f"INSERT INTO {postings_new} (file_ref, hash_code, time_offset) "
                    f"SELECT f.id, p.hash_code, p.time_offset "
                    f"FROM {self._postings_table} p JOIN {files_new} f ON f.file_id = p.file_id"
                )
                cur.execute(f"DROP TABLE {self._postings_table}")
                cur.execute(f"DROP TABLE {self._files_table}")
                cur.execute(f"ALTER TABLE {files_new} RENAME TO {self._files_table}")
                cur.execute(f"ALTER TABLE {postings_new} RENAME TO {self._postings_table}")
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
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._files_table}")
                count = int(cur.fetchone()[0])
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return count

    @property
    @_synchronized
    def posting_count(self) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._postings_table}")
                count = int(cur.fetchone()[0])
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return count

    @_synchronized
    def add(self, fingerprint: Fingerprint) -> None:
        self._record_format_version(fingerprint)
        self.remove(fingerprint.file_id)
        metadata = _index_metadata(fingerprint)
        with self._conn.cursor() as cur:
            # remove() above deleted any prior files row, so RETURNING id yields a
            # fresh surrogate; postings carry that file_ref, not the file_id string.
            cur.execute(
                f"INSERT INTO {self._files_table} (file_id, metadata) "
                "VALUES (%s, %s::jsonb) RETURNING id",
                (fingerprint.file_id, json.dumps(metadata, sort_keys=True)),
            )
            file_ref = cur.fetchone()[0]
            cur.executemany(
                f"INSERT INTO {self._postings_table} (file_ref, hash_code, time_offset) "
                "VALUES (%s, %s, %s)",
                [
                    (file_ref, self._encode(item.hash_code), int(item.time_offset))
                    for item in fingerprint.hashes
                ],
            )
        self._conn.commit()

    @_synchronized
    def add_many(self, fingerprints: Iterable[Fingerprint]) -> None:
        """Ingest the whole batch in ONE transaction, streaming postings via COPY.

        Equivalent to per-item :meth:`add`: each ``file_id`` is removed before it
        is (re)inserted, so a duplicate ``file_id`` keeps only the LAST
        fingerprint and any pre-batch rows for it are deleted once -- identical
        postings, metadata, and search results. Postings stream through
        ``cursor.copy`` (PostgreSQL's bulk path, far faster than row-at-a-time
        INSERT); metadata goes via ``executemany``; everything commits once.
        """

        # Fail-closed PRE-CHECK (no pin/persist) so a cross-format batch raises
        # before any write OR version stamp -- true all-or-nothing (F1).
        fingerprints = list(fingerprints)
        self._validate_batch_format(fingerprints)
        # Last-wins per file_id (a sequential add() of a repeated id removes the
        # earlier copy), which also avoids the files PRIMARY KEY conflict.
        survivors: dict[str, Fingerprint] = {}
        for fingerprint in fingerprints:
            self._record_format_version(fingerprint)
            survivors[fingerprint.file_id] = fingerprint
        if not survivors:
            return

        # Per file: metadata blob plus encoded postings; the surrogate file_ref
        # is filled in once the files row is inserted and its id known.
        staged: list[tuple[str, str, list[tuple[int, int]]]] = []
        for file_id, fingerprint in survivors.items():
            metadata = _index_metadata(fingerprint)
            staged.append((
                file_id,
                json.dumps(metadata, sort_keys=True),
                [(self._encode(item.hash_code), int(item.time_offset)) for item in fingerprint.hashes],
            ))

        file_ids = list(survivors)
        try:
            with self._conn.cursor() as cur:
                # remove() effect for every target file_id (pre-batch rows):
                # postings are keyed by the surrogate, so delete by file_ref of
                # the existing rows, then drop the files rows.
                cur.execute(
                    f"DELETE FROM {self._postings_table} WHERE file_ref IN "
                    f"(SELECT id FROM {self._files_table} WHERE file_id = ANY(%s))",
                    (file_ids,),
                )
                cur.execute(
                    f"DELETE FROM {self._files_table} WHERE file_id = ANY(%s)", (file_ids,)
                )
                # Insert each files row, capturing its fresh surrogate id, then
                # stage that file's postings with the resolved file_ref.
                posting_rows: list[tuple[int, int, int]] = []
                for file_id, meta_json, encoded in staged:
                    cur.execute(
                        f"INSERT INTO {self._files_table} (file_id, metadata) "
                        "VALUES (%s, %s::jsonb) RETURNING id",
                        (file_id, meta_json),
                    )
                    file_ref = cur.fetchone()[0]
                    posting_rows.extend((file_ref, code, offset) for code, offset in encoded)
                if posting_rows:
                    with cur.copy(
                        f"COPY {self._postings_table} (file_ref, hash_code, time_offset) FROM STDIN"
                    ) as copy:
                        for row in posting_rows:
                            copy.write_row(row)
        except BaseException:
            self._conn.rollback()  # all-or-nothing; never leave a partial ingest
            raise
        self._conn.commit()

    @_synchronized
    def remove(self, file_id: str) -> None:
        with self._conn.cursor() as cur:
            # Postings are keyed by the surrogate; delete by the file's file_ref
            # (resolved via subselect), then drop the files row.
            cur.execute(
                f"DELETE FROM {self._postings_table} WHERE file_ref = "
                f"(SELECT id FROM {self._files_table} WHERE file_id = %s)",
                (file_id,),
            )
            cur.execute(f"DELETE FROM {self._files_table} WHERE file_id = %s", (file_id,))
        self._conn.commit()

    @_synchronized
    def query(self, hash_code: int) -> list[IndexPosting]:
        code = int(hash_code)
        try:
            with self._conn.cursor() as cur:
                # JOIN to files to recover the str file_id from the surrogate.
                cur.execute(
                    f"SELECT f.file_id, p.time_offset FROM {self._postings_table} p "
                    f"JOIN {self._files_table} f ON f.id = p.file_ref WHERE p.hash_code = %s",
                    (self._encode(code),),
                )
                rows = cur.fetchall()
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return [
            IndexPosting(file_id=row[0], hash_code=code, time_offset=int(row[1]))
            for row in rows
        ]

    @_synchronized
    def query_many(self, hash_codes: Iterable[int]) -> dict[int, list[IndexPosting]]:
        codes = list({int(c) for c in hash_codes})
        results: dict[int, list[IndexPosting]] = {code: [] for code in codes}
        if not codes:
            return results
        encoded = [self._encode(code) for code in codes]
        try:
            with self._conn.cursor() as cur:  # single round-trip via array membership
                # JOIN to files to map each posting's surrogate back to file_id.
                cur.execute(
                    f"SELECT p.hash_code, f.file_id, p.time_offset FROM {self._postings_table} p "
                    f"JOIN {self._files_table} f ON f.id = p.file_ref "
                    "WHERE p.hash_code = ANY(%s)",
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

    @_synchronized
    def _aggregate(
        self,
        fingerprint: Fingerprint,
        offset_tolerance: int = 0,
        candidates: set[str] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Aggregate the offset histogram server-side in one round-trip.

        The query's (hash_code, offset) pairs are passed as two arrays and
        unnested into a derived table, joined to the postings, then grouped by
        (file_ref, delta) -- grouping on the small integer surrogate, not the
        64-char file_id. Only per-file aggregates return; the surrogate is
        resolved back to the original str ``file_id`` by a single JOIN to the
        files table in the final SELECT, so the result is keyed by the SAME
        ``file_id`` as every other backend.

        With ``offset_tolerance == 0`` (the default) the winning bin is picked
        server-side as votes DESC then delta ASC, matching the base in-memory
        tie-break -- unchanged and BYTE-IDENTICAL to before the option existed
        (partitioning by ``file_ref`` is a 1:1 relabelling of the file_id
        partitions, so the per-file winning row is unchanged). With ``> 0`` the
        compact per-(file, delta) histogram is returned and the banded winner is
        chosen by the shared :meth:`_banded_winner` so every backend bands
        identically.

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
                    SELECT p.file_ref AS file_ref, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                    FROM {self._postings_table} p JOIN q ON p.hash_code = q.hash_code
                ),
                bins AS (SELECT file_ref, delta, COUNT(*) AS votes FROM matches GROUP BY file_ref, delta),
                totals AS (
                    SELECT file_ref, COUNT(*) AS total_votes, COUNT(DISTINCT hc) AS uniq
                    FROM matches GROUP BY file_ref
                )
                SELECT f.file_id, b.delta, b.votes, t.total_votes, t.uniq
                FROM bins b JOIN totals t ON b.file_ref = t.file_ref
                JOIN {self._files_table} f ON f.id = b.file_ref
                """
        else:
            sql = f"""
                WITH q(hash_code, qoff) AS (SELECT * FROM unnest(%s::bigint[], %s::int[])),
                matches AS (
                    SELECT p.file_ref AS file_ref, (p.time_offset - q.qoff) AS delta, p.hash_code AS hc
                    FROM {self._postings_table} p JOIN q ON p.hash_code = q.hash_code
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
                JOIN {self._files_table} f ON f.id = r.file_ref
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

    @_synchronized
    def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
        file_total = self.file_count
        if file_total == 0:
            return 0
        threshold = max_df_ratio * file_total
        posts, files = self._postings_table, self._files_table
        with self._conn.cursor() as cur:
            # Document frequency is distinct surrogates touching a code (1:1 with
            # distinct file_ids), so COUNT(DISTINCT file_ref) is the same df test.
            cur.execute(
                f"DELETE FROM {posts} WHERE hash_code IN ("
                f"  SELECT hash_code FROM {posts} GROUP BY hash_code "
                f"  HAVING COUNT(DISTINCT file_ref) > %s)",
                (threshold,),
            )
            removed = cur.rowcount
            if removed:
                # Recalibrate stored hash_count to remaining postings per file,
                # joining the surrogate-keyed counts back to files by id.
                cur.execute(
                    f"UPDATE {files} f SET metadata = "
                    f"jsonb_set(f.metadata, '{{hash_count}}', to_jsonb(c.cnt)) "
                    f"FROM (SELECT file_ref, COUNT(*) AS cnt FROM {posts} GROUP BY file_ref) c "
                    f"WHERE f.id = c.file_ref"
                )
                cur.execute(
                    f"UPDATE {files} SET metadata = jsonb_set(metadata, '{{hash_count}}', '0'::jsonb) "
                    f"WHERE id NOT IN (SELECT DISTINCT file_ref FROM {posts})"
                )
        self._conn.commit()
        return removed

    @_synchronized
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

    @_synchronized
    def list_files(self) -> list[str]:
        try:
            with self._conn.cursor() as cur:
                # ORDER BY file_id matches the base sorted() contract server-side.
                cur.execute(f"SELECT file_id FROM {self._files_table} ORDER BY file_id")
                file_ids = [row[0] for row in cur.fetchall()]
        finally:
            self._conn.rollback()  # release the implicit read transaction even on error
        return file_ids

    @_synchronized
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

    @_synchronized
    def to_dict(self) -> dict[str, object]:
        files: dict[str, list[list[int]]] = {}
        metadata: dict[str, dict] = {}
        try:
            with self._conn.cursor() as cur:
                # Carry the surrogate id so postings can be read by file_ref; the
                # ORDER BY hash_code, time_offset is preserved so per-file posting
                # order is byte-identical to before.
                cur.execute(f"SELECT id, file_id FROM {self._files_table}")
                file_rows = cur.fetchall()
            for file_ref, file_id in file_rows:
                with self._conn.cursor() as cur:
                    cur.execute(
                        f"SELECT hash_code, time_offset FROM {self._postings_table} "
                        "WHERE file_ref = %s ORDER BY hash_code, time_offset",
                        (file_ref,),
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
