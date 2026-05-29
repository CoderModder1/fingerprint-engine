"""Command-line interface for the universal fingerprinting engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fingerprint_engine.core.fingerprinter import Fingerprinter
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

    add = subparsers.add_parser("add", help="Add one or more files to the index")
    add.add_argument("files", nargs="+")
    add.add_argument("--workers", type=int, default=None)

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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    fingerprinter = Fingerprinter(config_from_args(args))
    index_path = Path(args.index_path)

    if args.command == "fingerprint":
        fingerprint = fingerprinter.fingerprint_file(args.file)
        print(json.dumps(summarize_fingerprint(fingerprint, args.full), indent=2, sort_keys=True))
        return 0

    if args.command == "add":
        index = open_index(args)
        fingerprints = fingerprinter.fingerprint_many(args.files, max_workers=args.workers)
        for fingerprint in fingerprints:
            index.add(fingerprint)
        if args.backend == "memory":
            index.save(index_path)  # Redis/SQLite persist on add; no file to write.
        payload = {
            "backend": args.backend,
            "index_path": index_location(args),
            "indexed_files": [
                summarize_fingerprint(fingerprint, full=False) for fingerprint in fingerprints
            ],
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

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
