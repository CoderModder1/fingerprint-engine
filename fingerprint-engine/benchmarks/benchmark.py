"""Benchmark the fingerprint engine on a large real corpus.

Measures, across configurable corpus sizes and the in-memory + SQLite backends:
  * fingerprinting throughput (files/s, MB/s) over the whole corpus,
  * per-backend index-build throughput and on-disk/snapshot footprint,
  * query latency distribution (mean / p50 / p95 / max), isolating index lookup
    from fingerprinting by pre-computing the query fingerprints,
  * accuracy at scale: exact recall@1, near-duplicate (edited-file) recall@1,
    and the confidence separation between the true match and the best other.

By default it scans the running interpreter's stdlib + site-packages for source
files, so it is portable. Override by passing directories to scan.

Usage:
    python benchmarks/benchmark.py [ROOT ...] [--sizes 200,1000,2000]
                                   [--ext .py] [--min-bytes 1024] [--max-bytes 204800]
                                   [--query-sample 50]
Outputs a JSON report to stdout.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import statistics
import sys
import sysconfig
import tempfile
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")  # 0-hash RuntimeWarnings are expected for tiny inputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.fingerprinter import Fingerprinter
from core.index import InMemoryHashIndex, SQLiteHashIndex
from core.models import FingerprintConfig


def collect_files(roots: list[str], ext: str, min_bytes: int, max_bytes: int, limit: int) -> list[str]:
    paths: list[str] = []
    for root in roots:
        if not root:
            continue
        for path in glob.glob(f"{root}/**/*{ext}", recursive=True):
            try:
                if min_bytes <= os.path.getsize(path) <= max_bytes:
                    paths.append(path)
            except OSError:
                pass
    return sorted(set(paths))[:limit]


def query_stats(index, sample) -> dict:
    latencies, hits, self_conf, other_conf = [], 0, [], []
    for query in sample:
        start = time.perf_counter()
        results = index.search(query, top_k=3)
        latencies.append((time.perf_counter() - start) * 1000)
        if results and results[0].file_id == query.file_id:
            hits += 1
        self_conf.append(next((r.confidence for r in results if r.file_id == query.file_id), 0.0))
        other_conf.append(max((r.confidence for r in results if r.file_id != query.file_id), default=0.0))
    latencies.sort()
    return {
        "queries": len(sample),
        "mean_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(latencies[len(latencies) // 2], 3),
        "p95_ms": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 3),
        "max_ms": round(latencies[-1], 3),
        "recall_at_1": round(hits / len(sample), 4),
        "mean_self_confidence": round(statistics.mean(self_conf), 3),
        "mean_best_other_confidence": round(statistics.mean(other_conf), 4),
    }


def near_dup_recall(index, fingerprinter, sample, workdir: str) -> dict:
    """Append lines to each file, re-fingerprint, confirm it still finds the parent."""
    hits, confidences = 0, []
    edited = os.path.join(workdir, "edited.py")
    for query in sample:
        try:
            text = open(query.path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        with open(edited, "w", encoding="utf-8") as handle:
            handle.write(text + "\n# benchmark edit\n" * 5)
        results = index.search(fingerprinter.fingerprint_file(edited), top_k=1)
        if results and results[0].file_id == query.file_id:
            hits += 1
            confidences.append(results[0].confidence)
    return {
        "edited_queries": len(sample),
        "recall_at_1": round(hits / len(sample), 4),
        "mean_confidence": round(statistics.mean(confidences), 3) if confidences else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the fingerprint engine on a large corpus")
    parser.add_argument("roots", nargs="*", help="directories to scan (default: stdlib + site-packages)")
    parser.add_argument("--sizes", default="200,1000,2000", help="comma-separated corpus sizes")
    parser.add_argument("--ext", default=".py", help="file extension to scan for")
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=200 * 1024)
    parser.add_argument("--query-sample", type=int, default=50)
    args = parser.parse_args(argv)

    roots = args.roots or [sysconfig.get_paths().get("stdlib"), sysconfig.get_paths().get("purelib")]
    sizes = sorted(int(s) for s in args.sizes.split(",") if s.strip())
    paths = collect_files(roots, args.ext, args.min_bytes, args.max_bytes, max(sizes))
    if len(paths) < max(sizes):
        print(f"warning: only {len(paths)} files found for max size {max(sizes)}", file=sys.stderr)

    fingerprinter = Fingerprinter(FingerprintConfig())

    # Fingerprint the corpus once; this is the dominant, reusable cost.
    start = time.perf_counter()
    fingerprints, total_bytes, failures = [], 0, 0
    for path in paths:
        try:
            fingerprint = fingerprinter.fingerprint_file(path)
        except Exception:
            failures += 1
            continue
        fingerprints.append(fingerprint)
        total_bytes += fingerprint.size_bytes
    fp_secs = time.perf_counter() - start
    n = len(fingerprints)
    if not n:
        print("no files fingerprinted; pass corpus directories as arguments", file=sys.stderr)
        return 1
    total_mb = total_bytes / 1048576

    report = {
        "corpus": {"files": n, "failures": failures, "total_MB": round(total_mb, 1),
                   "roots": roots, "available_pool": len(paths)},
        "fingerprint_pass": {
            "seconds": round(fp_secs, 2),
            "files_per_sec": round(n / fp_secs, 1),
            "MB_per_sec": round(total_mb / fp_secs, 2),
            "total_hashes": sum(f.hash_count for f in fingerprints),
            "avg_hashes_per_file": round(sum(f.hash_count for f in fingerprints) / n, 1),
        },
        "sizes": [],
    }

    rng = random.Random(1234)
    workdir = tempfile.mkdtemp(prefix="fp_bench_")
    sizes = [s for s in sizes if s <= n]

    for size in sizes:
        subset = fingerprints[:size]
        sample = rng.sample(subset, min(args.query_sample, size))

        mem = InMemoryHashIndex()
        start = time.perf_counter()
        for fingerprint in subset:
            mem.add(fingerprint)
        mem_add = time.perf_counter() - start
        snapshot = os.path.join(workdir, "snap.json")
        mem.save(snapshot)
        snapshot_mb = os.path.getsize(snapshot) / 1048576

        db = os.path.join(workdir, "bench.sqlite3")
        for ext in ("", "-wal", "-shm"):
            try:
                os.remove(db + ext)
            except OSError:
                pass
        sqlite_index = SQLiteHashIndex(db)
        start = time.perf_counter()
        for fingerprint in subset:
            sqlite_index.add(fingerprint)
        sqlite_add = time.perf_counter() - start
        db_mb = os.path.getsize(db) / 1048576

        entry = {
            "corpus_files": size,
            "postings": mem.posting_count,
            "snapshot_MB": round(snapshot_mb, 1),
            "indexing": {
                "memory_add_files_per_sec": round(size / mem_add, 1),
                "sqlite_add_files_per_sec": round(size / sqlite_add, 1),
                "sqlite_db_MB": round(db_mb, 1),
            },
            "query_memory": query_stats(mem, sample),
            "query_sqlite": query_stats(sqlite_index, sample),
        }
        if size == sizes[-1]:
            entry["near_dup_memory"] = near_dup_recall(mem, fingerprinter, sample[:30], workdir)
        report["sizes"].append(entry)

        sqlite_index.close()
        del mem, sqlite_index
        gc.collect()

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
