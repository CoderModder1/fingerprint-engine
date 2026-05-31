# Benchmark results (current, post-optimization)

Raw data: [`baseline-results.json`](baseline-results.json). Reproduce with
`python benchmarks/benchmark.py --sizes 200,1000`.

- **Date:** 2026-05-29
- **Environment:** macOS (Darwin 25.x, Apple Silicon, arm64), CPython 3.13.13,
  single-threaded.
- **Code under test:** HEAD, i.e. *after* the three SQL-query optimizations
  (`d1bbdad` batched hash lookups, `ee5c303` server-side offset aggregation,
  `84d400d` stop-hash pruning). `prune_stop_hashes` is **not** applied in this
  run — `search()` is measured against the full, unpruned index, so these are a
  floor; enabling pruning lowers latency and storage further.
- **Corpus:** 1000 real Python source files (1–200 KB each), 17.9 MB total, from
  the interpreter stdlib + site-packages. 0 fingerprinting failures.
- **Scope note (bounded re-run):** the original baseline scanned up to 2000
  files; this refresh is capped at 200 and 1000 to keep wall-clock reasonable.
  The superseded pre-optimization 2000-file run is preserved verbatim in
  [`pre-optimization-results.json`](pre-optimization-results.json) for
  historical comparison. The 2000-file row in the tables below is therefore
  omitted rather than restated with stale numbers.

## Throughput & footprint

| corpus | postings | mem add (files/s) | sqlite add (files/s) | snapshot | sqlite DB |
|-------:|---------:|------------------:|---------------------:|---------:|----------:|
|    200 |    667 K |             212.7 |                 32.4 |   16.5 MB |  119 MB |
|  1 000 |   3.69 M |             305.9 |                 10.7 |   91.8 MB |  660 MB |

Fingerprinting (handler + FFT pipeline) ran at **30 files/s (0.54 MB/s)**,
averaging **~3 692 hashes/file** for source text (window 512). This pass is
unchanged by the optimizations (they touch index lookup, not fingerprinting).

## Query latency (50 queries; index lookup only, query fingerprints precomputed)

| corpus | mem mean / p50 / p95 | sqlite mean / p50 / p95 |
|-------:|---------------------:|------------------------:|
|    200 |   26 / 14 / 79 ms    |    192 / 150 / 398 ms    |
|  1 000 |  225 / 98 / 751 ms   |   1 252 / 828 / 3 130 ms |

The SQLite path is now dominated by one batched round-trip per search instead of
~3 700 single-row `SELECT`s. At 1000 files SQLite mean latency dropped from
**6 633 ms → 1 252 ms** (~5.3x) and p95 from **20 881 ms → 3 130 ms** (~6.7x)
relative to the pre-optimization baseline; the worst-case `max` fell from 35 s to
~5 s. (At 200 files the mean is flat — 158 ms → 192 ms — because at that scale
the per-query fixed cost dominates the savings, but tail latency still improved:
p95 482 → 398 ms, max 1 323 → 816 ms.) In-memory query is unchanged within
run-to-run noise, as expected — dict lookups never had the round-trip problem.

## Accuracy at scale

| corpus | recall@1 (exact) | self-conf | best-other-conf |
|-------:|-----------------:|----------:|----------------:|
|    200 |             0.96 |       1.0 |           0.193 |
|  1 000 |             1.00 |       1.0 |           0.045 |

**Near-duplicate recall@1 = 1.0** (mean confidence 0.98) at 1000 files: every
edited file still found its parent. Self-vs-best-other confidence separation is
large at every scale (1.0 vs ≤0.19). The 0.96 at 200 files is a tie-break
artifact — a couple of sampled files have a near-duplicate twin in the stdlib
that shares all their hashes (both confidence 1.0), so the lexicographically
smaller `file_id` wins. The optimizations are lossless: recall and confidence
match the pre-optimization baseline at every size.

## Findings

1. **Accuracy holds at scale and the SQL optimizations are lossless.** recall@1
   is 1.0 at 1k files, near-duplicate recall is 1.0, and confidence calibration
   keeps a wide true-vs-noise gap — identical to the pre-optimization run.
2. **In-memory query scales well** — sub-second median through 1000 files.
3. **SQLite query is no longer the headline bottleneck.** The batched
   `WHERE hash_code IN (...)` lookup plus server-side offset-histogram
   aggregation turned ~30 s/query (at 2000 files) and ~6.6 s/query (at 1000)
   into low-single-digit-second worst cases and a ~0.8 s median at 1000 files.
   Stop-hash pruning (`prune_stop_hashes`, not exercised here) trims this further
   by dropping the highest-document-frequency hashes before they reach the join.

### Remaining levers

Ranked by impact. **Ingest throughput and posting volume are now addressed in the
Tier-4 update below**; the SQL-query work above is already done.

- **Ingest throughput.** SQLite `add` is still the slowest stage: 10.7 files/s at
  1000 files (each `add` commits/fsyncs). Batching commits — or `COPY` for the
  Postgres backend — would raise build throughput; fingerprinting itself caps
  end-to-end ingest at ~30 files/s regardless.
- **In-memory footprint.** The snapshot is 91.8 MB at 1000 files and the live
  index holds 3.69 M postings; the resident set grows roughly linearly. Compact
  posting encodings (delta/varint offsets, interned file ids) would shrink both
  RSS and snapshot size.
- **Posting volume.** ~3 692 hashes/file → 3.69 M postings and a 660 MB SQLite DB
  at 1000 files; storage and latency both scale with this. Lowering
  `constellation_fanout` / `max_peaks_per_frame`, or applying stop-hash pruning
  by default, would cut volume at some recall cost — a tunable trade-off.
- **Approximate nearest-neighbour (ANN).** The current path is exact
  hash-intersection. For very large corpora an ANN/locality-sensitive layer over
  the constellation hashes would bound query cost sub-linearly, at the price of
  exactness.

These are single-machine numbers from a bounded (≤1000-file) re-run; treat them
as representative of the current code's relative behaviour, not as a hard
capacity ceiling.

## Tier-4 update (2026-05-30)

A performance pass landing four levers; **all are output-preserving** —
fingerprint hashes and search rankings are byte-identical, verified against the
prior commit (30 819 hashes over text/image/audio inputs). Bounded re-run at
sizes 200/500 (the slow SQLite query phase keeps full 2000-file runs
impractical); raw data in [`tier4-results.json`](tier4-results.json).

### Bulk ingest (`add_many`) — addresses the ingest-throughput lever

`HashIndex.add_many()` commits a whole batch in one transaction (SQLite also sets
`synchronous=NORMAL`; Redis pipelines; Postgres uses `COPY`) and is proven
identical to a sequential `add()` loop (same postings, metadata, and ranks,
including replace-existing semantics). The CLI `add` now uses it.

| corpus | per-file `add()` | `add_many()` | speedup |
|-------:|-----------------:|-------------:|--------:|
|    200 |      34.7 files/s |  103.9 files/s |  3.0× |
|    500 |      18.8 files/s |   82.0 files/s |  4.4× |

The speedup grows with corpus size as the per-`add` fsync cost is amortized.

### Faster fingerprinting (vectorized peak extraction)

`extract_peaks` now derives the clipped 3×3 local maximum as a separable,
`-inf`-padded numpy max instead of a per-candidate Python `neighborhood.max()` —
landmarks and hashes are byte-identical. The fingerprint pass rose from ~30 →
**43.9 files/s** on this site-packages corpus (larger on small/dense inputs).
Opt-in `fingerprint_many(executor="process")` additionally spreads the GIL-bound
hashing across cores for large batches.

### Posting-volume slack (fanout / max-peaks sweep)

Recall stays at **1.0** while hashes/file — and therefore storage and query
fan-in — fall by ~60%, confirming the defaults trade storage for headroom most
corpora don't need (a tunable `--fanout` / `--max-peaks-per-frame` lever):

| `fanout` / `max_peaks` | hashes/file | snapshot (120 files) | recall@1 |
|:----------------------:|------------:|---------------------:|:--------:|
| 6 / 8 (default)        |       3 725 |             11.11 MB |      1.0 |
| 4 / 5                  |       2 193 |              6.57 MB |      1.0 |
| 3 / 4                  |       1 455 |              4.39 MB |      1.0 |

### Memory

`IndexPosting`, `LandmarkPoint`, and `ConstellationHash` are now slotted
(`@dataclass(slots=True)`), dropping each instance's per-object `__dict__` — a
saving that compounds across millions of postings.

### Still open (deferred Tier-4 follow-ups)

Integer `file_id` surrogate key + compact posting encoding (footprint), SQL
`top_k`/`HAVING` query fan-out pushdown, ANN/LSH candidate generation, and
streaming ingest of very large files.
