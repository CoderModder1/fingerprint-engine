"""Command-line interface for the universal fingerprinting engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.fingerprinter import Fingerprinter
from core.index import HashIndex, InMemoryHashIndex, RedisHashIndex
from core.models import Fingerprint, FingerprintConfig


DEFAULT_INDEX_PATH = Path(__file__).with_name(".fingerprint_index.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal file fingerprinting engine")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="JSON index path")
    parser.add_argument("--window-size", type=int, default=FingerprintConfig.window_size)
    parser.add_argument("--hop-size", type=int, default=FingerprintConfig.hop_size)
    parser.add_argument("--peak-threshold", type=float, default=FingerprintConfig.peak_threshold)
    parser.add_argument("--peak-percentile", type=float, default=FingerprintConfig.peak_percentile)
    parser.add_argument("--fanout", type=int, default=FingerprintConfig.constellation_fanout)
    parser.add_argument("--hash-bits", type=int, default=FingerprintConfig.hash_bits)
    parser.add_argument("--min-time-frames", type=int, default=FingerprintConfig.min_time_frames)
    parser.add_argument("--min-window-size", type=int, default=FingerprintConfig.min_window_size)
    parser.add_argument("--backend", choices=("memory", "redis"), default="memory",
                        help="index backend (default: memory)")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0",
                        help="Redis connection URL (used when --backend redis)")
    parser.add_argument("--redis-prefix", default="fpidx",
                        help="Redis key namespace (used when --backend redis)")

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

    return parser


def config_from_args(args: argparse.Namespace) -> FingerprintConfig:
    return FingerprintConfig(
        window_size=args.window_size,
        hop_size=args.hop_size,
        peak_threshold=args.peak_threshold,
        peak_percentile=args.peak_percentile,
        constellation_fanout=args.fanout,
        hash_bits=args.hash_bits,
        min_time_frames=args.min_time_frames,
        min_window_size=args.min_window_size,
    )


def open_index(args: argparse.Namespace) -> HashIndex:
    """Open the index for the selected backend. Memory loads from --index-path;
    Redis connects to a live, already-persistent store."""

    if getattr(args, "backend", "memory") == "redis":
        return RedisHashIndex(url=args.redis_url, key_prefix=args.redis_prefix)
    return InMemoryHashIndex.load(Path(args.index_path))


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
            index.save(index_path)  # Redis persists on add; no file to write.
        payload = {
            "backend": args.backend,
            "index_path": str(index_path) if args.backend == "memory" else args.redis_url,
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
        results = index.search(query, top_k=args.top_k)
        payload = {
            "query": summarize_fingerprint(query, full=False),
            "backend": args.backend,
            "index_path": str(index_path) if args.backend == "memory" else args.redis_url,
            "results": [result.to_dict() for result in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
