# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Tier-1 through Tier-5 hardening: correctness, durability, reliability,
performance, and the product/operability layer. With ONE exception called out
below (the v2 fingerprint format, which changes audio derivation), none of this
changes the default fingerprint derivation or search ranking -- see
`VERSIONING.md` for the stability guarantees and the path to 1.0.

### Changed — BREAKING: fingerprint format v2 (re-index required)

`FINGERPRINT_FORMAT_VERSION` is now `2`. A v1 index must be rebuilt; the
format-version check detects a v1 query against a v2 index. Two derivation
changes, both verified to leave the SEVEN non-audio handlers' hash codes
byte-identical (only the version stamp advances for them) while audio changes:

- **Cross-platform determinism.** The signal/spectrogram reductions
  (`mean`/`std`/`percentile`) now accumulate in **float64** instead of float32.
  float32 reductions drift ~1e-7 with reduction order / SIMD width / numpy
  version, which on a near-zero-mean signal (cancellation) can shift the
  normalisation and flip a borderline peak across platforms. Output is identical
  for non-audio handlers on real inputs; it corrects audio's near-zero-mean
  normalisation (so audio hash codes change). `np.percentile` is pinned to
  `method="linear"`.
- **Audio excerpt/clip matching by default.** The audio handler now fingerprints
  with a multi-resolution window bank `(512, 1024, 2048, 4096)` by default
  (`AudioFileHandler.default_window_bank`), so excerpt/clip recall works out of
  the box (previously ~0 at a single window; a documented limitation) at ~N× the
  audio postings. A global `FingerprintConfig.window_bank` still overrides it and
  an explicit `--window-size` disables it. (The float64 fix already recovers
  excerpt recall on stationary audio; the bank is kept on by default for
  robustness on real non-stationary audio.)

### Tests / CI

- The "all four backends rank identically" guarantee is now proven for Postgres
  in CI: the workflow runs a live Postgres 16 service container with
  `FINGERPRINT_TEST_PG_DSN` set and `psycopg` installed (via the `[dev]`/`[all]`
  extras), so the gated `@requires_pg` parity suite executes for real on every
  push. Added Postgres parity coverage for `candidate_limit` (the shared-posting
  prefilter) and a Postgres concurrency test that exercises the `@_synchronized`
  per-index lock under concurrent `add`/`search` on one shared connection.

### Added

- Exception hierarchy in `fingerprint_engine.core.exceptions`: a base
  `FingerprintError` with `NoHandlerError`, `MissingDependencyError`,
  `FileTooLargeError`, and `InvalidSnapshotError` (the last also subclasses
  `ValueError` so existing `except ValueError` snapshot guards keep working).
  All are re-exported from the top-level `fingerprint_engine` package.
- Public API surface: `fingerprint_engine` now re-exports `Fingerprinter`,
  `FingerprintConfig`, `Calibration`, `Fingerprint`, `SearchResult`,
  `LandmarkPoint`, `ConstellationHash`, the `HashIndex` contract and all four
  backends (`InMemoryHashIndex`, `SQLiteHashIndex`, `RedisHashIndex`,
  `PostgresHashIndex`), the dedup types/`find_duplicates`, and the exception
  classes, with an `__all__` and a `__version__` resolved via
  `importlib.metadata`.
- Snapshot `schema_version` field (currently `1`) with validation on load, so
  an unsupported or structurally invalid snapshot fails loudly with
  `InvalidSnapshotError` instead of loading partial or misinterpreted state.
- Atomic, durable index snapshots: `save()` writes to a temporary file and
  `os.replace`s it into place, keeping a `.bak` of the previous snapshot;
  `_read_snapshot` recovers from the `.bak` if the primary is corrupt.
- `HashIndex` enumeration contract: `list_files()`, `iter_metadata()`,
  `contains()`/`__contains__`, and bulk `add_many()`/`query_many()`, uniform
  across all four backends and surfaced by the CLI `list` command.
- CLI `list` command (enumerate indexed files, `--summary` for counts only),
  `dedup` command (exact + near-duplicate clustering over given paths), and
  `--incremental`/`--skip-existing` ingest (skip files whose content sha is
  already indexed, fingerprinting only the misses).
- CLI `doctor` command: a deps health check that reports the Python and
  `fingerprint-engine` versions and, for each optional extra
  (image/audio/pdf/video/embeddings/redis/postgres/service), which of its
  required vs. optional dependencies import and which handlers/backends/encoders
  are therefore available (the embedding handler's precomputed-vector path is
  numpy-only, so it is reported under core; `audio` is available with scipy/WAV
  alone, with pydub/MP3 reported as an optional capability). Pure introspection;
  always exits 0.
- Stateless FastAPI service (`fingerprint_engine.service`) behind the `service`
  extra, reusing the engine verbatim (no scoring/ranking reimplemented).
- Process-pool execution mode for `fingerprint_many` (`executor="process"`),
  opt-in and byte-identical to thread mode -- only *where* the CPU-bound work
  runs changes, never the output.
- Resource limits for untrusted input (`max_file_size_bytes`, `max_pdf_pages`)
  with finite defaults that bound the OOM/DoS vector; see `SECURITY.md`.
- Fast, deterministic accuracy harness (`benchmarks/accuracy.py` +
  `tests/test_accuracy.py`) and a large-corpus throughput/footprint/latency
  benchmark suite.
- Apache License 2.0 (`LICENSE` + `NOTICE`), declared via the PEP 639 SPDX
  `license = "Apache-2.0"` expression and `license-files`, plus trove
  classifiers in `pyproject.toml`.
- `VERSIONING.md` documenting the semantic-versioning policy, the stable public
  interfaces (snapshot schema, `HashIndex` contract, `SearchResult`/confidence
  semantics, fingerprint derivation) and their compatibility guarantees, and
  the 1.0 definition-of-done checklist.

### Changed

- Missing optional dependencies now fail loud: a handler whose required extra is
  not installed raises `MissingDependencyError` (carrying the package and extra
  name) instead of silently demoting to the binary fallback and producing
  fingerprints that are incomparable to a properly-installed install.
- SQLite/Postgres `search()` aggregate the offset histogram server-side in SQL
  and batch hash lookups (`query_many`), cutting round-trips; SQLite no longer
  holds a write lock while reading and supports use as a context manager
  (`__enter__`/`__exit__`). Postgres operations roll back on error to avoid
  leaving an aborted transaction open on the shared connection.
- Ingest is transactional/bulk per batch: one commit (SQLite) / one pipeline
  (Redis) / one COPY (Postgres) instead of a per-file write.
- Peak extraction is vectorized and `IndexPosting` is slotted, reducing
  per-file CPU and per-posting memory; results are unchanged.
- `prune_stop_hashes` drops non-discriminative high document-frequency codes
  and recalibrates stored hash counts, cutting query latency and storage. This
  is an explicit, caller-invoked operation that does not run by default.
- Batch fingerprinting (`fingerprint_many`) is fail-soft: one bad file no longer
  aborts the whole batch; failures are reported per file and the rest succeed.
- Dev dependencies are sourced solely from the `[dev]` extra in
  `pyproject.toml`. The drifted, duplicated `requirements.txt` (which had lost
  `ruff` and `mypy`) was removed; use `pip install -e ".[dev]"`.

### Fixed

- README "Tests" section no longer hardcodes a machine-specific absolute path;
  it documents `pip install -e ".[dev]"` and running `pytest` from the repo root.

### Security

Hardening from an adversarial review. None of these change the default fingerprint
derivation or search ranking (verified byte-identical old-vs-new on a
text/image/audio/archive corpus across the SQLite and in-memory backends).

- Cross-format index integrity now FAILS CLOSED: adding a fingerprint whose hash
  format version differs from the index's pinned version raises
  `FormatVersionMismatchError` instead of only warning and committing the
  incompatible postings (which silently mixed two hash code spaces and could
  fabricate matches). Opt into the legacy warn-and-proceed behaviour with
  `HashIndex.allow_format_mixing = True`. `add_many` validates the whole batch
  up front, so a rejected cross-format batch leaves the index — and its durable
  version stamp — completely untouched (all-or-nothing).
- The corpus hash-format version is now persisted by the durable backends
  (SQLite/Postgres side meta table, Redis key) and restored on open, so reopening
  a non-default-config store no longer silently reports the engine baseline and
  bypasses the cross-format compatibility check. A default corpus records nothing
  (absent meta == baseline), keeping the default store byte-identical.
- SQLite (and Postgres) backends serialize all connection access under a
  per-index re-entrant lock. Previously concurrent `search()` calls — as the
  FastAPI service issues from its threadpool against one shared index — could
  interleave the connection-global `_query` temp-table DELETE/INSERT/SELECT and
  return cross-contaminated rankings or raise `InterfaceError`.
- The HTTP service enforces `max_file_size_bytes` DURING upload streaming: an
  oversized body is rejected (413) after at most the limit reaches temp disk,
  instead of being spooled in full before the engine's post-write size check.
- The archive handler bounds decompression work against zip/tar bombs: an
  aggregate decompressed-read budget (`max_total_content_bytes`), a total-entry
  cap counting every entry including non-file tar entries (`max_entries`), and a
  tar traversal bound. Over-budget members degrade to their CRC/size identity
  token (never raising). Normal archives are unaffected.
- Routing correctness: `.npz` files (numpy vector containers, which are zip
  archives) are no longer claimed by the generic archive handler off the zip
  magic. The archive handler declines `.npz` so it routes to `EmbeddingFileHandler`
  and gets the advertised vector-sequence fingerprint instead of a structural
  archive one. Ordinary `.zip`/`.tar`/`.tar.gz` routing is unchanged.

#### Second adversarial audit (2026-05-31)

A second deep multi-agent audit (14 verified findings). All fixed; default
fingerprint hash codes verified byte-identical old-vs-new across ALL eight routed
handlers (text, binary, image, audio/WAV, archive, embedding `.npy`/`.jsonl`,
video).

- **Single-read pipeline (identity TOCTOU).** `fingerprint_file` read each file
  twice — once for `content_sha256`/`file_id`, then again inside the handler's
  `load()` — so a concurrent writer between the reads could make the stored
  identity describe different bytes than the hashes were derived from (silent
  dedup/identity corruption), and the second read bypassed `max_file_size_bytes`.
  `FileHandler.load` now takes the already-read bytes; the fingerprinted bytes
  are provably the bytes the identity describes.
- **Durable `save()`.** An empty (zero-file) snapshot saved over a populated
  primary clobbered the only recoverable copy (and, on a second save, the `.bak`
  too); it is now refused with `SnapshotWriteRefused` unless `force=True`.
  Concurrent same-process saves shared a PID-only temp name and raced
  (`FileNotFoundError`/lost write); the temp is now uniquely named per writer.
  The `.bak` is `fsync`-ed before the primary is replaced.
- **SQLite `add()`** now rolls back on failure (no committed phantom zero-posting
  file), and `ConstellationHash` rejects hash codes outside the unsigned-64 range
  at construction so every backend behaves identically (the SQL backends would
  otherwise overflow while in-memory silently accepted).
- **`RedisHashIndex`** mutators (`add`/`add_many`/`remove`) now serialize under a
  per-index re-entrant lock, matching the SQL backends' concurrency contract
  (the multi-step read-modify-write otherwise double-counted postings under the
  shared service threadpool).
- **`candidate_limit`** prefilter now ranks on shared-POSTING count (a true upper
  bound on aligned votes) instead of distinct-code count, so a tight limit can no
  longer drop a genuine match dominated by a code repeated at coherent offsets.
- **Snapshot load** counts and warns on dropped malformed/out-of-range postings,
  and `from_dict` recomputes `hash_count` from the postings actually loaded so a
  degraded snapshot's match confidence stays calibrated.
- **Image decode bomb:** new `max_image_pixels` (default ~89.5 Mpx; `0` =
  unlimited) rejects an over-cap image from its header BEFORE decode, so a tiny
  highly-compressible file can no longer decode to hundreds of megapixels.
- Smaller robustness fixes: `file_content_sha256` enforces a running byte cap and
  rejects non-regular files; WEBP is recognised by its magic bytes; MP3
  de-interleaving tolerates a non-frame-aligned sample count; and the CLI `add`
  counts block reconciles (`scanned == skipped_existing + newly_indexed +
  failed`).

## [0.1.0] - 2026-05-29

Baseline release.

### Added

- Shazam-style universal file fingerprinting: per-handler 1D signal extraction,
  FFT-equivalent spectrogram, landmark peak picking, constellation pairing, and
  deterministic integer hash codes.
- File handlers for binary, text/source, image, audio (WAV plus MP3 via
  `pydub`/ffmpeg), and PDF inputs, auto-discovered as plugins and ranked by
  `can_handle()` score and priority.
- Storage-agnostic `HashIndex` contract with four backends sharing one
  time-coherent, offset-histogram `search()`: in-memory (JSON-persisted),
  SQLite, Redis, and Postgres.
- Handler-independent match `confidence` in [0, 1] alongside the raw `score`,
  with uniform and per-handler `Calibration` thresholds.
- Stop-hash pruning (`prune_stop_hashes`) to drop non-discriminative high
  document-frequency codes and recalibrate stored hash counts.
- Adaptive windowing so signals shorter than `window_size` still produce a
  usable fingerprint, recording the effective window/hop per file.
- OPT-IN multi-resolution window bank (`FingerprintConfig.window_bank`,
  default `None` = off and byte-identical to the single-window path). When set
  to a small tuple of window sizes, a signal is fingerprinted once per window,
  with the window folded into each hash so window-w codes only collide with
  window-w codes; a query at the same bank can align at whatever resolution
  survives a cross-length transform. This fixes the audio excerpt/clip recall
  that single-window matching scores at ~0 (measured 0.0 -> 1.0 on the accuracy
  harness for both `clip_prefix_60pct` and `excerpt_mid`). Cost: a bank of N
  windows multiplies a file's postings by roughly N (measured ~2.7x text,
  ~3.4x audio, ~8.3x image), so the bank is bounded by `max_window_bank_size`
  (default 6) and is a recall/storage trade, not a default.
- Scale-invariance design: fixed per-handler windows for sequence handlers and
  canonical 256x256 grayscale normalisation for images.
- `fingerprint-engine` CLI (`fingerprint`, `add`, `search`, `prune`) with
  configurable resolution and backend selection.
- Benchmark harness covering fingerprinting throughput, per-backend build rate
  and footprint, query latency, scaling, and accuracy at scale.
