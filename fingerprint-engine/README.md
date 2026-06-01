# Universal File Fingerprinting Engine

A modular file fingerprinting engine inspired by Shazam's landmark-based audio matching. Each file type is converted into a 1D signal, transformed into a spectrogram-like matrix, reduced to landmark peaks, paired into constellations, and indexed as compact searchable hashes.

## Architecture

The engine has three layers:

1. Core orchestration
   - `fingerprint_engine/core/fingerprinter.py` discovers `FileHandler` plugins from the `fingerprint_engine/handlers` package.
   - `fingerprint_engine/core/models.py` defines `Fingerprint`, `LandmarkPoint`, `ConstellationHash`, and tuning config dataclasses.
   - `fingerprint_engine/core/index/` defines the storage-agnostic `HashIndex` contract plus the four index backends.

2. FFT-equivalent pipeline
   - `fingerprint_engine/core/fft_pipeline.py` normalizes handler signals, applies sliding windows, runs `numpy.fft.rfft`, extracts adaptive local maxima, builds peak-pair constellations, and hashes `(freq1, freq2, delta_t)` into deterministic integer codes.
   - Binary files use normalized raw bytes.
   - Text and source files use character code, character class, and token-length rhythm.
   - Images use flattened grayscale pixel intensity.
   - Audio uses decoded mono samples for WAV, and MP3 via `pydub`/ffmpeg when available.
   - PDFs use extracted page text plus page boundary markers.
   - Video uses a sequence of canonical keyframes sampled at a fixed temporal cadence (the `[video]` extra: PyAV + Pillow).
   - Embeddings fingerprint an ordered *sequence* of precomputed dense vectors (`.npy`/`.npz`/`.jsonl`, numpy-only); raw text can be encoded on load via the `[embeddings]` extra (model2vec).

3. Scalability boundaries
   - Handlers are auto-discovered plugins, not selected by hardcoded type switches in the core.
   - Fingerprint resolution is configurable from Python or the CLI.
   - Batch fingerprinting uses `ThreadPoolExecutor`.
   - `HashIndex` can be replaced with Redis, SQLite, Postgres, or another backend without changing handlers or the pipeline.

## Install

```bash
cd fingerprint-engine
python -m pip install -e ".[all]"     # all handlers + backends; installs the `fingerprint-engine` CLI
# or pick extras (see the matrix below): pip install -e ".[image,audio,pdf]"
# or just the core (numpy only): pip install -e .
```

This installs the `fingerprint-engine` console command. For development without
installing, run the CLI as `python -m fingerprint_engine.cli ...`.

numpy is the only hard runtime dependency; every other capability is behind an
optional extra, lazily imported (a core-only install never pulls them in). Run
`fingerprint-engine doctor` to see exactly what is available in your environment.

| Extra | Unlocks | Installs |
|-------|---------|----------|
| `image` | image handler (raster + opt-in pHash) | Pillow |
| `audio` | audio handler — WAV (scipy) + MP3 (pydub/ffmpeg) | scipy, pydub, `audioop-lts` (Python ≥ 3.13) |
| `pdf` | PDF handler | pypdf |
| `video` | video keyframe handler | PyAV, Pillow |
| `embeddings` | encode-on-load encoder for the embedding handler | model2vec (static, no torch) |
| `redis` | Redis index backend | redis |
| `postgres` | Postgres index backend | psycopg |
| `service` | HTTP service (`fingerprint_engine.service`) | fastapi, uvicorn, python-multipart |
| `all` | every runtime extra above | — |

MP3 support requires the `[audio]` extra and a working ffmpeg install. On
**Python 3.13** `pydub` additionally needs the `audioop-lts` backport (PEP 594
removed the stdlib `audioop` module); the `[audio]` extra pins it automatically
via an environment marker. The embedding handler's precomputed-vector path
(`.npy`/`.npz`/`.jsonl`) is numpy-only and always available — the `[embeddings]`
extra is only for encoding raw text on load. **Embedding matching finds shared
*exact* vector sub-sequences (a near-duplicate stream), not semantic /
nearest-neighbour similarity** — a full paraphrase shares no vectors and will not
match.

## CLI Usage

Fingerprint a file:

```bash
fingerprint-engine fingerprint path/to/file
```

Add files to the default local JSON index:

```bash
fingerprint-engine add path/to/file1 path/to/file2
```

Search the index with a query file:

```bash
fingerprint-engine search path/to/query --top-k 5
```

Use a custom index path:

```bash
fingerprint-engine --index-path ./my-index.json add path/to/file
fingerprint-engine --index-path ./my-index.json search path/to/query
```

Tune resolution:

```bash
fingerprint-engine --window-size 2048 --hop-size 512 --fanout 8 fingerprint path/to/file
```

## Python Usage

```python
from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import FingerprintConfig

fingerprinter = Fingerprinter(FingerprintConfig(window_size=2048, hop_size=512))
index = InMemoryHashIndex()

fingerprint = fingerprinter.fingerprint_file("document.pdf")
index.add(fingerprint)

results = index.search(fingerprinter.fingerprint_file("query.pdf"))
```

## Search Model

Search uses time-coherent matching:

1. Query hashes are looked up in the index.
2. Each posting votes for a candidate file and offset delta.
3. Candidate scores are dominated by the strongest offset histogram bin.
4. Results are ranked by aligned votes, unique hash coverage, and total match quality.

This mirrors the core Shazam idea: many weak hash matches become strong evidence only when they agree on a consistent relative offset.

## Match Confidence & Calibration

The raw `score` scales with a file's hash count, so its magnitude is not
comparable across handlers (a PDF self-match scores ~400 while an image scores
~3500). Each `SearchResult` therefore also carries a **handler-independent
`confidence` in [0, 1]** — the fraction of the *smaller* fingerprint's hashes
that aligned at the winning offset. On a real corpus, true matches land at
0.5–1.0 across every handler while unrelated files sit below ~0.02, so one
threshold separates them all:

```bash
fingerprint-engine search query.pdf --min-confidence 0.05   # drop matches below 0.05
```

```python
from fingerprint_engine.core.models import Calibration
# Uniform threshold (usually enough, since confidence is already normalised):
index.search(query, calibration=Calibration(default_min_confidence=0.05))
# Per-handler overrides when a type needs a stricter/looser cutoff:
index.search(query, calibration=Calibration(
    default_min_confidence=0.05, per_handler={"text": 0.10}))
```

Ranking still uses the raw `score` (correct within a single query); `confidence`
is the comparable measure for accept/reject decisions across handlers.

## Stop-Hash Pruning

Query cost and storage are dominated by posting volume: a query touches every
posting of each of its hash codes, and common-but-non-discriminative codes
(present in many files) carry most of the postings. `prune_stop_hashes(max_df_ratio)`
removes postings for any hash code present in more than `max_df_ratio` of the
indexed files, and recalibrates each file's stored `hash_count` so confidence
stays meaningful (a self-match remains ~1.0).

```bash
fingerprint-engine --backend sqlite --sqlite-path index.sqlite3 prune --max-df-ratio 0.1
```

On a 1000-file source corpus (default `0.1`), this removed ~36% of postings and
made in-memory queries **~5× faster** (≈0.05 for ~51% / ~10×), with **recall@1
and self-confidence unchanged at 1.0** — the pruned codes are noise, not signal.
Supported by the in-memory, SQLite, and Postgres backends (Redis raises
`NotImplementedError`; rebuild it from a pruned snapshot instead). You can also
generate fewer hashes up front via `--fanout` / `--max-peaks-per-frame`.

## Adaptive Windowing

A signal shorter than `window_size` collapses to one or two FFT frames, so no
constellation pair can span `min_delta_t` and the fingerprint would be empty
(and therefore unsearchable). To prevent that, the pipeline shrinks the window
toward `min_time_frames` (default 16) frames whenever the configured window is
too large for the input — preserving the configured window:hop overlap ratio
and never going below `min_window_size` (default 16). Long signals are
untouched, so normal-length inputs keep identical fingerprints.

Consequences worth knowing:

- The window actually used is recorded per file in
  `fingerprint.metadata["effective_window_size"]` / `["effective_hop_size"]`,
  so adaptation is never silent.
- Because matching compares fingerprints produced with the same window,
  identical content (identical length) always uses the same window and matches.
  Adaptation also affects medium files (signals between `window_size` and
  roughly `window_size + (min_time_frames - 1) * hop_size` samples), giving them
  richer fingerprints than a single near-empty frame would.
- Inputs too small to reach `min_time_frames` even at `min_window_size` fall
  back to the floor; if that still yields zero hashes, `Fingerprinter` emits a
  `RuntimeWarning` that the file is unsearchable.

Tune this behaviour with `--min-time-frames` / `--min-window-size` (CLI) or the
matching `FingerprintConfig` fields.

## Scale Invariance

Matching only works when the query and the indexed file share the same effective
FFT window — a fingerprint built with a 512-point window cannot align with one
built at 1024. Two design choices keep that window stable across size changes:

- **Fixed per-handler windows.** Sequence handlers (text, binary, PDF) declare a
  small fixed window (`default_signal_window = 512`) instead of inheriting the
  audio-tuned 4096 default. So a file and a *truncated copy / excerpt* of it use
  the same window and still match (their hashes are a subset relation), rather
  than landing on different length-adaptive windows. **Audio** fingerprints with
  a multi-resolution **window bank `(512, 1024, 2048, 4096)` by default** (as of
  fingerprint format **v2**), so audio excerpt/clip matching works out of the box
  (recall@1 ~1.0) — the smallest window lets an excerpt align to its parent. This
  was a known limitation of the former single-window mode (global signal
  normalisation and the global peak threshold both shift over a sub-segment); the
  v2 float64-accumulated reductions plus the default bank resolve it, at ~N× the
  audio postings. See `benchmarks/accuracy.py`, which measures it.
- **Canonical image normalisation.** Every image is resampled to a fixed
  256×256 grayscale grid before the signal is built, so the *same picture at a
  different resolution* (or after lossy re-encoding) maps to a comparable signal.
  A raw flattened-pixel signal would otherwise be destroyed by any resize.

Explicitly setting `--window-size` / `--hop-size` overrides the per-handler
windows globally (useful for experiments; all files then share one window).

What is **not** supported (and is outside the Shazam model): true geometric or
time-stretch invariance — e.g. cropping/rotating an image, or time-stretching
audio — because those change the underlying signal sequence, not just its scale.

## Tuning: opt-in matching settings

The defaults are tuned for exact/near-duplicate detection. **Audio** additionally
matches excerpts/clips out of the box (a multi-resolution window bank is the
audio default as of format **v2**). The other matching refinements below ship
**opt-in and off by default**; enable them per use-case. The numbers are from the
reproducible harness in `benchmarks/accuracy.py` (run
`python benchmarks/accuracy.py --mode hard`).

> **Hash compatibility:** settings marked *hash-changing* alter the derived
> hashes, so an index built with them is **not** comparable to a default index —
> you must re-index, and the query must use the same setting. The engine stamps a
> `fingerprint_format_version` and **warns (or raises, with `strict_format=True`)
> on a query/index format mismatch**, so a mismatch can't silently return wrong
> matches. Settings marked *search-time* leave hashes untouched (no re-index).

| Goal | Setting | Effect (hard corpus) | Cost / caveat | Kind |
|------|---------|----------------------|---------------|------|
| **Audio excerpt / clip matching** | *on by default* (audio uses `default_window_bank=(512,1024,2048,4096)`) | excerpt/clip recall@1 **~1.0** | ~N× more audio hashes/file; included in the v2 default | *default (v2)* |
| **Window bank for ALL handlers** | `FingerprintConfig(window_bank=(512, 1024, 2048, 4096))` | extends multi-resolution matching to text/binary/etc. (overrides the per-handler audio default globally) | ~N× more hashes/file — up to ~8× for images; index + query must share the bank | *hash-changing* |
| **Fuzzy / multi-edit text near-dups** | `search(query, offset_tolerance=1)` (or `Calibration(offset_tolerance=1)`) | multi-edit recall 0.83 → 0.90; scatter 0.90 → 0.97 | small, no measured precision loss | *search-time* |
| **Image resize / crop / rotate robustness** | `FingerprintConfig(image_mode="phash")` | crop/rotate/jpeg recall up vs the raster default | weaker impostor separation — **use a stricter cutoff (~0.2–0.3, not 0.05)** via `Calibration(per_handler={"image_phash": 0.25})` | *hash-changing* |
| **Bound query cost on a large corpus** | `search(query, candidate_limit=N)` | lossless when `N` ≥ true match set; sub-linear candidate scan | a tight `N` on a dense corpus can drop low-overlap tail matches | *search-time* |
| Spectral-shift tolerance (niche) | `FingerprintConfig(freq_quantization=2)` | raises confidence on shifted spectra | **net-negative recall + worse precision on general near-dups — not recommended as a blanket setting** | *hash-changing* |

> **pHash is its own handler.** `image_mode="phash"` selects a distinct handler
> (the raster handler stays the default and claims images under `image_mode="raster"`;
> the two are mutually exclusive). Fingerprints made in phash mode are labelled
> `handler="image_phash"`, so the stricter cutoff is keyed on **`"image_phash"`**
> (not `"image"`) — `Calibration(per_handler={"image_phash": 0.25})` — and the raster
> `"image"` operating point is untouched. The recommended cutoff is also exposed as
> `IMAGE_PHASH_RECOMMENDED_MIN_CONFIDENCE` (0.25) / `IMAGE_PHASH_RECOMMENDED_CALIBRATION_KEY`
> in `fingerprint_engine.handlers.image_phash_handler`.

```python
# Audio excerpt/clip matching is ON BY DEFAULT (v2) -- no config needed:
from fingerprint_engine.core.fingerprinter import Fingerprinter
fingerprinter = Fingerprinter()                 # audio uses its default window bank

# To extend the multi-resolution bank to EVERY handler (text/binary/...), set it
# globally (re-index required; the query must use the same config):
from fingerprint_engine.core.models import FingerprintConfig
global_bank_cfg = FingerprintConfig(window_bank=(512, 1024, 2048, 4096))

# Example: a resize/crop/rotate-tolerant image index using the dedicated pHash
# handler, with the stricter cutoff it needs keyed on its own handler name.
from fingerprint_engine.core.models import Calibration
phash_cfg = FingerprintConfig(image_mode="phash")        # routes images to "image_phash"
phash_fp = Fingerprinter(phash_cfg)                      # index + queries both use phash_cfg
calibration = Calibration(per_handler={"image_phash": 0.25})  # stricter cutoff, raster untouched
```

## Extension Guide

### Add A File Handler

Create a module in `fingerprint_engine/handlers/` containing a subclass of `FileHandler`:

```python
from pathlib import Path
import numpy as np
from fingerprint_engine.handlers.base import FileHandler

class MyFormatHandler(FileHandler):
    name = "my_format"
    priority = 80
    supported_extensions = {".mine"}

    def load(self, path: str | Path):
        return Path(path).read_bytes()

    def to_signal(self, payload) -> np.ndarray:
        return np.asarray(payload, dtype=np.uint8).astype(np.float32)
```

The orchestrator imports handler modules dynamically and ranks them by `can_handle()` score and priority.

### Index Backends

The engine ships four backends. All implement the storage-agnostic `HashIndex`
contract and **share the same `search()`, `save()`, and `load_snapshot()`** (the
time-coherent offset-histogram scoring lives in the base class), so they rank
identically and scores stay comparable:

- **`InMemoryHashIndex`** — dict-backed, JSON-persisted. The default.
- **`SQLiteHashIndex`** — file-persistent, zero extra dependencies (stdlib
  `sqlite3`), indexed lookups. Good single-node persistence.
- **`RedisHashIndex`** — postings live in Redis, persistent and shareable across
  processes (horizontal scale). Requires `redis` (and a running server); tests
  use `fakeredis`, no server needed.
- **`PostgresHashIndex`** — server-grade shared/durable store; postings in an
  indexed table, metadata as `JSONB`. Requires `psycopg` and a running server;
  the gated integration tests run when `FINGERPRINT_TEST_PG_DSN` is set, which CI
  does against a live Postgres service container on every push.

All four backends produce **identical** `SearchResult` rankings for the same
index contents and query (one shared scoring path on the base class). This is
enforced by cross-backend parity tests — in-memory/SQLite/Redis run everywhere,
and Postgres parity (search ranking, tie-breaks, `add_many`, snapshot interop,
the surrogate key, `candidate_limit`, and concurrent access) is proven by the
`@requires_pg` suite against the live Postgres service in CI.

```bash
# CLI: pick a backend with --backend (default: memory)
fingerprint-engine --backend sqlite   --sqlite-path index.sqlite3 add path/to/file
fingerprint-engine --backend redis    --redis-url redis://localhost:6379/0 add path/to/file
fingerprint-engine --backend postgres --postgres-dsn postgresql://localhost/fingerprint add path/to/file
fingerprint-engine --backend postgres --postgres-dsn postgresql://localhost/fingerprint search path/to/query
```

```python
from fingerprint_engine.core.index import SQLiteHashIndex, RedisHashIndex, PostgresHashIndex
index = SQLiteHashIndex("index.sqlite3")            # or ":memory:"; or inject a sqlite3.Connection
# index = RedisHashIndex(url="redis://localhost:6379/0")      # or inject a client (e.g. fakeredis)
# index = PostgresHashIndex(dsn="postgresql://localhost/fingerprint")  # or inject a psycopg connection
index.add(fingerprint)
results = index.search(query_fingerprint)
index.save("snapshot.json")                          # portable export (interops across backends)
SQLiteHashIndex(":memory:").load_snapshot("snapshot.json")  # bulk-load any snapshot
```

> Note: both SQLite and PostgreSQL store hash codes as *signed* 64-bit integers,
> so the unsigned 64-bit codes are mapped reversibly into signed range
> (`code - 2**63`) to avoid overflow.

To add your own backend, subclass `HashIndex` and implement `add(fingerprint)`,
`remove(file_id)`, `query(hash_code)`, `_metadata_for(file_id)`, and `to_dict()`.
Keep `query(hash_code)` returning postings with `file_id`, `hash_code`, and
`time_offset`; the inherited `search()`/`save()`/`load_snapshot()` do the rest.

## Tests

Install the dev extra (handlers, backends, and the test/lint/type-check tools —
this is the single source of truth for dev dependencies; there is no
`requirements.txt`), then run the suite from the repo root:

```bash
pip install -e ".[dev]"
pytest
```

`ruff check .` and `mypy fingerprint_engine` are also part of the dev toolchain
and are installed by the `[dev]` extra.

## Benchmark

`benchmarks/benchmark.py` measures fingerprinting throughput, per-backend index
build rate and footprint, query-latency distribution, scaling across corpus
sizes, and accuracy at scale (exact / near-duplicate recall@1 and confidence
separation). By default it scans the running interpreter's stdlib + site-packages
for a large real corpus; pass directories to scan your own.

```bash
python benchmarks/benchmark.py                      # default corpus + sizes
python benchmarks/benchmark.py /path/to/files --sizes 500,5000   # custom
```
