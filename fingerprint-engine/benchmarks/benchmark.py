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
                                   [--query-sample 50] [--fanout-sweep [N]]
Outputs a JSON report to stdout.

The SQLite ingest measurement reports BOTH the legacy per-file ``add()`` rate and
the bulk ``add_many()`` rate (one transaction for the whole batch). ``add_many``
is the path production ingest should use; the two are output-equivalent (same
postings, metadata, and search results -- see ``HashIndex.add_many``), differing
only in commit/fsync amortization.

``--fanout-sweep`` additionally fingerprints a fixed sample of the corpus at a
few ``(constellation_fanout, max_peaks_per_frame)`` settings and reports
hashes/file, recall@1, and snapshot footprint at each -- illustrating the
recall-vs-storage trade-off. It is OFF by default so the standard run is
unchanged.
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

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex, SQLiteHashIndex
from fingerprint_engine.core.models import FingerprintConfig


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


def _fresh_sqlite(db: str) -> SQLiteHashIndex:
    """Open a fresh SQLite index, removing any prior db/WAL/SHM files first."""

    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass
    return SQLiteHashIndex(db)


def sqlite_ingest_rates(subset: list, workdir: str) -> dict:
    """Measure SQLite ingest two ways on the SAME corpus: per-file add vs add_many.

    The two are output-equivalent (HashIndex.add_many is documented to produce
    the same postings/metadata/search results as sequential add()); this isolates
    the commit/fsync amortization win of the bulk path. ``files`` count is the
    distinct file_ids actually persisted (the corpus may contain content
    duplicates that share a file_id, so add()'s remove-then-insert collapses them
    -- using the persisted count keeps the rate honest and the two paths
    comparable).
    """

    size = len(subset)

    per_file_db = os.path.join(workdir, "ingest_addper.sqlite3")
    index = _fresh_sqlite(per_file_db)
    start = time.perf_counter()
    for fingerprint in subset:
        index.add(fingerprint)
    add_secs = time.perf_counter() - start
    persisted = index.file_count
    add_db_mb = os.path.getsize(per_file_db) / 1048576
    index.close()

    bulk_db = os.path.join(workdir, "ingest_addmany.sqlite3")
    index = _fresh_sqlite(bulk_db)
    start = time.perf_counter()
    index.add_many(subset)
    add_many_secs = time.perf_counter() - start
    bulk_db_mb = os.path.getsize(bulk_db) / 1048576
    index.close()

    return {
        "submitted_files": size,
        "persisted_files": persisted,
        "add_per_file_files_per_sec": round(persisted / add_secs, 1) if add_secs else 0.0,
        "add_many_files_per_sec": round(persisted / add_many_secs, 1) if add_many_secs else 0.0,
        "speedup": round(add_secs / add_many_secs, 1) if add_many_secs else 0.0,
        "add_per_file_db_MB": round(add_db_mb, 1),
        "add_many_db_MB": round(bulk_db_mb, 1),
    }


def fanout_sweep(paths: list[str], settings: list[tuple[int, int]], sample_size: int,
                 query_sample: int, workdir: str) -> list[dict]:
    """Fingerprint a fixed sample at several (fanout, max_peaks) settings.

    Reports hashes/file, exact recall@1, and snapshot footprint at each setting
    so the recall-vs-storage slack is visible. The same ``paths`` sample is used
    for every setting so rows are directly comparable. Only window-independent
    knobs (constellation_fanout, max_peaks_per_frame) are varied, so the
    per-handler fixed-window grid is preserved and self-matches stay aligned.
    """

    sample_paths = paths[:sample_size]
    rng = random.Random(99)
    rows: list[dict] = []
    for fanout, max_peaks in settings:
        config = FingerprintConfig(constellation_fanout=fanout, max_peaks_per_frame=max_peaks)
        fingerprinter = Fingerprinter(config)
        fingerprints = []
        for path in sample_paths:
            try:
                fingerprints.append(fingerprinter.fingerprint_file(path))
            except Exception:
                continue
        if not fingerprints:
            continue
        index = InMemoryHashIndex()
        index.add_many(fingerprints)
        snapshot = os.path.join(workdir, f"sweep_{fanout}_{max_peaks}.json")
        index.save(snapshot)
        snapshot_mb = os.path.getsize(snapshot) / 1048576
        query = rng.sample(fingerprints, min(query_sample, len(fingerprints)))
        hits = sum(
            1
            for q in query
            if (r := index.search(q, top_k=1)) and r[0].file_id == q.file_id
        )
        total_hashes = sum(f.hash_count for f in fingerprints)
        rows.append({
            "constellation_fanout": fanout,
            "max_peaks_per_frame": max_peaks,
            "files": len(fingerprints),
            "avg_hashes_per_file": round(total_hashes / len(fingerprints), 1),
            "postings": index.posting_count,
            "snapshot_MB": round(snapshot_mb, 2),
            "recall_at_1": round(hits / len(query), 4),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the fingerprint engine on a large corpus")
    parser.add_argument("roots", nargs="*", help="directories to scan (default: stdlib + site-packages)")
    parser.add_argument("--sizes", default="200,1000,2000", help="comma-separated corpus sizes")
    parser.add_argument("--ext", default=".py", help="file extension to scan for")
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=200 * 1024)
    parser.add_argument("--query-sample", type=int, default=50)
    parser.add_argument(
        "--fanout-sweep",
        nargs="?",
        type=int,
        const=200,
        default=0,
        help="run a (fanout, max_peaks) sweep on N sampled files (default 200 when given as a bare flag)",
    )
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
                # Bulk-ingest path (one transaction): the production-recommended
                # route and the headline ingest win. Output-equivalent to the
                # per-file add() above (same postings/metadata/results).
                "sqlite_ingest": sqlite_ingest_rates(subset, workdir),
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

    if args.fanout_sweep:
        report["fanout_sweep"] = fanout_sweep(
            paths,
            settings=[(6, 8), (4, 5), (3, 4)],
            sample_size=min(args.fanout_sweep, n),
            query_sample=args.query_sample,
            workdir=workdir,
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
