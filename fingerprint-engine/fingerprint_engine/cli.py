"""Command-line interface for the universal fingerprinting engine."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from fingerprint_engine import __version__
from fingerprint_engine.core.dedup import DEFAULT_MIN_CONFIDENCE, find_duplicates
from fingerprint_engine.core.exceptions import (
    FileTooLargeError,
    FingerprintError,
    MissingDependencyError,
    NoHandlerError,
)
from fingerprint_engine.core.fingerprinter import (
    Fingerprinter,
    expand_paths,
    file_content_sha256,
)
from fingerprint_engine.core.index import (
    HashIndex,
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from fingerprint_engine.core.models import Calibration, Fingerprint, FingerprintConfig

DEFAULT_INDEX_PATH = Path(__file__).with_name(".fingerprint_index.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal file fingerprinting engine")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="JSON index path")
    parser.add_argument("--window-size", type=int, default=FingerprintConfig.window_size)
    parser.add_argument("--hop-size", type=int, default=FingerprintConfig.hop_size)
    parser.add_argument("--peak-threshold", type=float, default=FingerprintConfig.peak_threshold)
    parser.add_argument("--peak-percentile", type=float, default=FingerprintConfig.peak_percentile)
    parser.add_argument("--fanout", type=int, default=FingerprintConfig.constellation_fanout)
    parser.add_argument("--max-peaks-per-frame", type=int, default=FingerprintConfig.max_peaks_per_frame,
                        help="cap landmark peaks per frame (lower = fewer hashes/file)")
    parser.add_argument("--hash-bits", type=int, default=FingerprintConfig.hash_bits)
    parser.add_argument("--min-time-frames", type=int, default=FingerprintConfig.min_time_frames)
    parser.add_argument("--min-window-size", type=int, default=FingerprintConfig.min_window_size)
    parser.add_argument("--max-file-size", type=int, default=FingerprintConfig.max_file_size_bytes,
                        help="reject input files larger than this many bytes before reading them "
                             "(0 = unlimited); bounds the OOM vector from untrusted input")
    parser.add_argument("--max-pdf-pages", type=int, default=FingerprintConfig.max_pdf_pages,
                        help="cap PDF pages decoded per file (0 = unlimited)")
    parser.add_argument("--backend", choices=("memory", "redis", "sqlite", "postgres"),
                        default="memory", help="index backend (default: memory)")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0",
                        help="Redis connection URL (used when --backend redis)")
    parser.add_argument("--redis-prefix", default="fpidx",
                        help="Redis key namespace (used when --backend redis)")
    parser.add_argument("--sqlite-path", default=".fingerprint_index.sqlite3",
                        help="SQLite database path (used when --backend sqlite)")
    parser.add_argument("--postgres-dsn", default="postgresql://localhost/fingerprint",
                        help="PostgreSQL connection string (used when --backend postgres)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint = subparsers.add_parser("fingerprint", help="Fingerprint a file")
    fingerprint.add_argument("file")
    fingerprint.add_argument("--full", action="store_true", help="Emit full hashes and landmarks")

    add = subparsers.add_parser(
        "add", help="Add files and/or directories (walked recursively) to the index"
    )
    add.add_argument("files", nargs="+", help="files and/or directories to ingest")
    add.add_argument("--workers", type=int, default=None)
    add.add_argument(
        "--incremental", "--skip-existing", dest="incremental", action="store_true",
        help="skip files whose content is already indexed (matched cheaply by "
             "sha256 of the file bytes, without fingerprinting); only new/changed "
             "files are fingerprinted and added",
    )

    search = subparsers.add_parser("search", help="Search the index with a query file")
    search.add_argument("file")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--min-confidence", type=float, default=None,
                        help="drop matches below this confidence in [0,1] "
                             "(handler-comparable; e.g. 0.05)")

    prune = subparsers.add_parser("prune", help="Remove non-discriminative 'stop' hash codes")
    prune.add_argument("--max-df-ratio", type=float, default=0.1,
                       help="prune hash codes present in more than this fraction of files "
                            "(default 0.1; lower = more aggressive)")

    list_files = subparsers.add_parser("list", help="List the files indexed in the selected backend")
    list_files.add_argument("--summary", action="store_true",
                            help="emit only the counts, omitting the per-file list")

    dedup = subparsers.add_parser(
        "dedup", help="Find exact and near-duplicate files among the given paths"
    )
    dedup.add_argument("files", nargs="+")
    dedup.add_argument("--workers", type=int, default=None)
    dedup.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                       help="near-duplicate confidence cutoff in [0,1] "
                            f"(default {DEFAULT_MIN_CONFIDENCE}; higher = stricter)")

    subparsers.add_parser(
        "doctor",
        help="Report Python/engine versions and which optional extras (and the "
             "handlers/backends they enable) are importable in this environment",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> FingerprintConfig:
    return FingerprintConfig(
        window_size=args.window_size,
        hop_size=args.hop_size,
        peak_threshold=args.peak_threshold,
        peak_percentile=args.peak_percentile,
        constellation_fanout=args.fanout,
        max_peaks_per_frame=args.max_peaks_per_frame,
        hash_bits=args.hash_bits,
        min_time_frames=args.min_time_frames,
        min_window_size=args.min_window_size,
        max_file_size_bytes=args.max_file_size,
        max_pdf_pages=args.max_pdf_pages,
    )


def open_index(args: argparse.Namespace) -> HashIndex:
    """Open the index for the selected backend. Memory loads from --index-path;
    Redis/SQLite connect to a live, already-persistent store."""

    backend = getattr(args, "backend", "memory")
    if backend == "redis":
        return RedisHashIndex(url=args.redis_url, key_prefix=args.redis_prefix)
    if backend == "sqlite":
        return SQLiteHashIndex(database=args.sqlite_path)
    if backend == "postgres":
        return PostgresHashIndex(dsn=args.postgres_dsn)
    return InMemoryHashIndex.load(Path(args.index_path))


def index_location(args: argparse.Namespace) -> str:
    """Human-readable location of the index for the selected backend."""

    if args.backend == "redis":
        return args.redis_url
    if args.backend == "sqlite":
        return args.sqlite_path
    if args.backend == "postgres":
        return args.postgres_dsn
    return str(Path(args.index_path))


def summarize_fingerprint(fingerprint: Fingerprint, full: bool = False) -> dict[str, Any]:
    if full:
        return fingerprint.to_dict(include_landmarks=True)
    return {
        "file_id": fingerprint.file_id,
        "path": fingerprint.path,
        "handler": fingerprint.handler,
        "size_bytes": fingerprint.size_bytes,
        "content_sha256": fingerprint.content_sha256,
        "landmark_count": fingerprint.landmark_count,
        "hash_count": fingerprint.hash_count,
        "sample_hashes": [
            {"hash_code": item.hash_code, "time_offset": item.time_offset}
            for item in fingerprint.hashes[:10]
        ],
        "metadata": fingerprint.metadata,
    }


# Optional extras: which import probe(s) gate each extra, and what
# handlers/backends become usable once it is installed. Kept as data so the
# `doctor` report and the diagnosis stay in lock-step with pyproject's
# [project.optional-dependencies]. Each entry maps the extra name to the
# importable module names it requires (ALL must import for the extra to be
# usable) and the human-readable capabilities it unlocks. The core install
# (numpy only) always provides the binary/text/archive handlers and the
# in-memory + SQLite backends, reported separately under "core".
_EXTRA_PROBES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("image", ("PIL",), ("handler:image",)),
    # The audio handler needs scipy for WAV and pydub (+ffmpeg) for MP3; both
    # ship in the single `audio` extra, so the handler is only fully usable when
    # both import.
    ("audio", ("scipy", "pydub"), ("handler:audio",)),
    ("pdf", ("pypdf",), ("handler:pdf",)),
    ("redis", ("redis",), ("backend:redis",)),
    ("postgres", ("psycopg",), ("backend:postgres",)),
    # python-multipart is required by Starlette to parse uploads; its canonical
    # import name is ``python_multipart`` (the legacy ``multipart`` alias emits a
    # PendingDeprecationWarning), so probe the canonical name.
    ("service", ("fastapi", "uvicorn", "python_multipart"), ("service:http",)),
)

# Capabilities available with only the core runtime dependency (numpy).
_CORE_HANDLERS: tuple[str, ...] = ("binary", "text", "archive")
_CORE_BACKENDS: tuple[str, ...] = ("memory", "sqlite")


def _probe_module(module: str) -> dict[str, Any]:
    """Attempt to import ``module``; report success and (on failure) the reason.

    Read-only: the import is the diagnosis. Any import-time failure (the package
    is absent, or present but broken) is caught and surfaced as ``ok: False`` with
    the error string, so ``doctor`` never raises -- it reports.
    """

    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - doctor reports every failure mode
        return {"module": module, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"module": module, "ok": True}


def _doctor_report() -> dict[str, Any]:
    """Build the environment diagnosis emitted by the ``doctor`` command.

    Reports the interpreter and engine versions, the always-available core
    capabilities, and, for each optional extra, whether its dependency imports
    succeed and which handlers/backends are therefore available. Pure
    introspection -- it imports candidate modules to test availability but never
    fingerprints, opens an index, or mutates anything.
    """

    extras: dict[str, Any] = {}
    available_handlers: list[str] = list(_CORE_HANDLERS)
    available_backends: list[str] = list(_CORE_BACKENDS)
    for extra, modules, capabilities in _EXTRA_PROBES:
        probes = [_probe_module(module) for module in modules]
        available = all(probe["ok"] for probe in probes)
        handlers = sorted(
            cap.split(":", 1)[1] for cap in capabilities if cap.startswith("handler:")
        )
        backends = sorted(
            cap.split(":", 1)[1] for cap in capabilities if cap.startswith("backend:")
        )
        services = sorted(
            cap.split(":", 1)[1] for cap in capabilities if cap.startswith("service:")
        )
        if available:
            available_handlers.extend(handlers)
            available_backends.extend(backends)
        extras[extra] = {
            "available": available,
            "requires": list(modules),
            "probes": probes,
            "handlers": handlers,
            "backends": backends,
            "services": services,
        }
    return {
        "python_version": platform.python_version(),
        "fingerprint_engine_version": __version__,
        "core": {
            "requires": ["numpy"],
            "handlers": list(_CORE_HANDLERS),
            "backends": list(_CORE_BACKENDS),
        },
        "extras": extras,
        "available_handlers": sorted(set(available_handlers)),
        "available_backends": sorted(set(available_backends)),
    }


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run the selected command. Raises library exceptions to ``main`` to be
    mapped onto clean stderr messages and distinct exit codes."""

    if args.command == "doctor":
        # Pure environment diagnosis: no Fingerprinter, no index, no I/O on the
        # corpus. Always succeeds (exit 0) -- unavailable extras are reported,
        # not errors.
        print(json.dumps(_doctor_report(), indent=2, sort_keys=True))
        return 0

    fingerprinter = Fingerprinter(config_from_args(args))
    index_path = Path(args.index_path)

    if args.command == "fingerprint":
        fingerprint = fingerprinter.fingerprint_file(args.file)
        print(json.dumps(summarize_fingerprint(fingerprint, args.full), indent=2, sort_keys=True))
        return 0

    if args.command == "add":
        index = open_index(args)
        # Expand directories into their files (recursive walk) and keep plain
        # files as-is, fail-soft: a missing/unreadable argument is recorded as a
        # skip rather than aborting the whole ingest. Sorted + de-duplicated, so
        # a directory always yields the same ordered file list.
        expand_errors: list[tuple[str, Exception]] = []
        targets = expand_paths(args.files, errors=expand_errors)
        skipped = [
            {"path": path, "reason": f"{type(exc).__name__}: {exc}"} for path, exc in expand_errors
        ]
        # `scanned` counts the TRUE input population so the counts block reconciles
        # (scanned == skipped_existing + newly_indexed + failed). expand_paths
        # diverts unexpandable arguments (missing/unreadable) into expand_errors,
        # which are counted in `failed`; including them here keeps the identity.
        scanned = len(targets) + len(expand_errors)
        skipped_existing = 0

        # Incremental ingest: before fingerprinting, drop files whose content is
        # already in the index. Since file_id == content_sha256, "already
        # indexed" means the file's content sha is a member of the index. We
        # compute that sha by reading the bytes (chunked) -- far cheaper than the
        # FFT fingerprint -- and only fingerprint the misses. A file whose sha
        # cannot be computed (vanished/oversized/unreadable) is recorded as a
        # skip here, never fingerprinted, mirroring the fail-soft contract.
        if getattr(args, "incremental", False):
            limit = fingerprinter.config.max_file_size_bytes
            misses: list[Path] = []
            for target in targets:
                try:
                    sha = file_content_sha256(target, max_file_size_bytes=limit)
                except (OSError, FileTooLargeError) as exc:
                    skipped.append(
                        {"path": str(target), "reason": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
                if index.contains(sha):
                    skipped_existing += 1
                else:
                    misses.append(target)
            to_fingerprint: list[Path] = misses
        else:
            to_fingerprint = list(targets)

        # Fail-soft batch: per-file errors are reported, not fatal, so a partial
        # batch still indexes (and saves) its successes. fingerprint_many collects
        # each failure as a structured (path, exc) tuple, so a path containing
        # ": " can never mis-split (unlike parsing the warning text).
        collector: list[tuple[str, Exception]] = []
        fingerprints = fingerprinter.fingerprint_many(
            to_fingerprint, max_workers=args.workers, errors=collector
        )
        skipped.extend(
            {"path": path, "reason": f"{type(exc).__name__}: {exc}"} for path, exc in collector
        )
        # Bulk/transactional ingest: one commit (SQLite) / one pipeline (Redis) /
        # one COPY (Postgres) for the whole batch instead of a per-file commit.
        # Equivalent to calling add() per fingerprint in sequence.
        index.add_many(fingerprints)
        if args.backend == "memory":
            index.save(index_path)  # Redis/SQLite persist on add; no file to write.
        payload = {
            "backend": args.backend,
            "index_path": index_location(args),
            "indexed_files": [
                summarize_fingerprint(fingerprint, full=False) for fingerprint in fingerprints
            ],
            "skipped": skipped,
            "counts": {
                # These reconcile exactly: scanned == skipped_existing +
                # newly_indexed + failed. Every entry in `skipped` is a genuine
                # failure (a bad/unreadable argument, an un-sha-able file, or a
                # fingerprint error); already-indexed files are counted in
                # skipped_existing instead and never appear in `skipped`.
                "scanned": scanned,
                "skipped_existing": skipped_existing,
                "newly_indexed": len(fingerprints),
                "failed": len(skipped),
            },
            "file_count": index.file_count,
            "posting_count": index.posting_count,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "search":
        index = open_index(args)
        query = fingerprinter.fingerprint_file(args.file)
        calibration = (
            Calibration(default_min_confidence=args.min_confidence)
            if args.min_confidence is not None
            else None
        )
        results = index.search(query, top_k=args.top_k, calibration=calibration)
        payload = {
            "query": summarize_fingerprint(query, full=False),
            "backend": args.backend,
            "index_path": index_location(args),
            "results": [result.to_dict() for result in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "prune":
        index = open_index(args)
        removed = index.prune_stop_hashes(max_df_ratio=args.max_df_ratio)
        if args.backend == "memory":
            index.save(index_path)
        payload = {
            "backend": args.backend,
            "index_path": index_location(args),
            "max_df_ratio": args.max_df_ratio,
            "pruned_postings": removed,
            "file_count": index.file_count,
            "posting_count": index.posting_count,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "list":
        index = open_index(args)
        # Stream per-file metadata rather than the heavy to_dict(); each dict is
        # projected down to the stable list fields. iter_metadata() yields in the
        # backend-independent sorted file_id order.
        files = [
            {
                "file_id": str(meta.get("file_id", "")),
                "path": str(meta.get("path", "")),
                "handler": str(meta.get("handler", "")),
                "hash_count": int(meta.get("hash_count", 0) or 0),
            }
            for meta in index.iter_metadata()
        ]
        payload = {
            "backend": args.backend,
            "index_path": index_location(args),
            "file_count": index.file_count,
            "posting_count": index.posting_count,
        }
        if not args.summary:
            payload["files"] = files
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "dedup":
        # Fingerprint each input once (fail-soft: bad paths are reported, not
        # fatal), then cluster. Dedup analyses the GIVEN paths against each
        # other in a scratch in-memory index -- it never reads or mutates the
        # selected persistent backend, so exact dupes (which the backend would
        # collapse to one entry) are detected from the inputs as designed.
        dedup_errors: list[tuple[str, Exception]] = []
        fingerprints = fingerprinter.fingerprint_many(
            args.files, max_workers=args.workers, errors=dedup_errors
        )
        skipped = [
            {"path": path, "reason": f"{type(exc).__name__}: {exc}"} for path, exc in dedup_errors
        ]
        report = find_duplicates(fingerprints, min_confidence=args.min_confidence)
        payload = {
            "min_confidence": args.min_confidence,
            "skipped": skipped,
            **report.to_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def main(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch, mapping library exceptions onto clean one-line
    stderr messages and distinct exit codes (instead of tracebacks):

    * ``2``  -- argparse usage errors (handled by argparse itself).
    * ``3``  -- :class:`MissingDependencyError` (optional dependency not installed).
    * ``4``  -- backend/connection errors (Redis/Postgres connect, missing index,
      or other operational ``RuntimeError``/``OSError`` from the index layer).
    * ``1``  -- input errors on the query/fingerprint path (no handler, missing
      file, a directory passed where a file was expected, or an input exceeding
      the configured ``--max-file-size``).
    * ``0``  -- success.

    ``SystemExit``/``KeyboardInterrupt`` are never caught here.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(parser, args)
    except MissingDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (NoHandlerError, FileTooLargeError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FingerprintError as exc:
        # Any other library error (e.g. an invalid/corrupt index snapshot) is a
        # backend/operational failure rather than a usage error.
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except (RuntimeError, OSError) as exc:
        # Backend connection/operation errors: Redis/psycopg connect failures, a
        # missing or unreadable index store, etc.
        print(f"error: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
