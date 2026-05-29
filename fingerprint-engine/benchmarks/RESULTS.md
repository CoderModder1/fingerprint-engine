# Benchmark results (baseline)

Raw data: [`baseline-results.json`](baseline-results.json). Reproduce with
`python benchmarks/benchmark.py`.

- **Date:** 2026-05-29
- **Environment:** macOS (Darwin, Apple Silicon), Python 3.13, single-threaded.
- **Corpus:** 2000 real Python source files (1–200 KB each), 34.5 MB total, from
  the interpreter stdlib + site-packages. 0 fingerprinting failures.

## Throughput & footprint

| corpus | postings | mem add (files/s) | sqlite add (files/s) | snapshot | sqlite DB |
|-------:|---------:|------------------:|---------------------:|---------:|----------:|
|    200 |    669 K |             103.5 |                 29.5 |   16.6 MB |  119 MB |
|  1 000 |   3.66 M |             274.1 |                  9.3 |   91.1 MB |  653 MB |
|  2 000 |   7.24 M |             187.7 |                  5.6 |  180.1 MB | 1.29 GB |

Fingerprinting (handler + FFT pipeline) ran at **28 files/s (0.49 MB/s)**,
averaging **~3 640 hashes/file** for source text (window 512).

## Query latency (50 queries; index lookup only, query fingerprints precomputed)

| corpus | mem mean / p50 / p95 | sqlite mean / p50 / p95 |
|-------:|---------------------:|------------------------:|
|    200 |   25 / 12 / 77 ms    |    158 / 79 / 483 ms    |
|  1 000 |  191 / 114 / 467 ms  |   6 633 / 4 070 / 20 881 ms |
|  2 000 |  319 / 108 / 1 069 ms | 30 696 / 9 965 / 95 816 ms |

## Accuracy at scale

| corpus | recall@1 (exact) | self-conf | best-other-conf |
|-------:|-----------------:|----------:|----------------:|
|    200 |             0.96 |       1.0 |           0.185 |
|  1 000 |             1.00 |       1.0 |           0.053 |
|  2 000 |             1.00 |       1.0 |           0.039 |

**Near-duplicate recall@1 = 1.0** (mean confidence 0.982) at 2000 files: every
edited file still found its parent. Self-vs-best-other confidence separation is
large at every scale (1.0 vs ≤0.19). The 0.96 at 200 files is a tie-break
artifact — a couple of sampled files have a near-duplicate twin in the stdlib
that shares all their hashes (both confidence 1.0), so the lexicographically
smaller `file_id` wins.

## Findings

1. **Accuracy holds at scale.** recall@1 is 1.0 at 1k/2k files, near-duplicate
   recall is 1.0, and the confidence calibration keeps a wide true-vs-noise gap.
2. **In-memory query scales acceptably** — sub-second median through 2000 files.
3. **SQL backends are query-bound (the main bottleneck).** `search()` issues one
   `query(hash_code)` per query hash — ~3 640 single-row `SELECT`s per search —
   so SQLite degrades to ~30 s/query at 2000 files. The fix is a **batched
   lookup** (`WHERE hash_code IN (...)` or a temp-table join) so a search is one
   round-trip; this would also benefit Postgres. (In-memory/Redis are unaffected:
   dict lookups / pipeline-able.)
4. **Storage is dominated by hash volume.** ~3 640 hashes/file → 7.2 M postings
   and a 1.3 GB SQLite DB at 2000 files. Lowering `constellation_fanout` /
   `max_peaks_per_frame` would cut storage and latency at some recall cost.
5. **SQLite ingest degrades** (per-`add` `commit()` fsync): batching commits
   (or `COPY` for Postgres) would raise build throughput.

These are single-machine baselines; the headline next optimization is batched
hash lookup for the SQL backends.
