# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Tier-1 through Tier-5 hardening: correctness, durability, reliability,
performance, and the product/operability layer. None of this changes the
default fingerprint derivation or search ranking -- see `VERSIONING.md` for the
stability guarantees and the path to 1.0.

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
  (image/audio/pdf/redis/postgres/service), whether its dependency imports
  and which handlers/backends are therefore available. Pure introspection;
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
