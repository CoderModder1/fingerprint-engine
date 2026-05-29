# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Tier-1 and Tier-2 hardening of correctness, durability, and reliability.

### Added

- Exception hierarchy in `fingerprint_engine.core.exceptions`: a base
  `FingerprintError` with `NoHandlerError`, `MissingDependencyError`, and
  `InvalidSnapshotError` (the last also subclasses `ValueError` so existing
  `except ValueError` snapshot guards keep working). All are re-exported from
  the top-level `fingerprint_engine` package.
- Public API surface: `fingerprint_engine` now re-exports `Fingerprinter`,
  `FingerprintConfig`, `Calibration`, `Fingerprint`, `SearchResult`,
  `LandmarkPoint`, `ConstellationHash`, the `HashIndex` contract and all four
  backends (`InMemoryHashIndex`, `SQLiteHashIndex`, `RedisHashIndex`,
  `PostgresHashIndex`), and the exception classes, with an `__all__` and a
  `__version__` resolved via `importlib.metadata`.
- Snapshot `schema_version` field with validation on load, so an unsupported or
  structurally invalid snapshot fails loudly with `InvalidSnapshotError` instead
  of loading partial or misinterpreted state.
- Atomic, durable index snapshots: `save()` writes to a temporary file and
  `os.replace`s it into place, keeping a `.bak` of the previous snapshot;
  `_read_snapshot` recovers from the `.bak` if the primary is corrupt.
- Documented concurrency contract for the index backends.
- Apache License 2.0 (`LICENSE` + `NOTICE`), declared via the PEP 639 SPDX
  `license = "Apache-2.0"` expression and `license-files`, plus trove
  classifiers in `pyproject.toml`.

### Changed

- Missing optional dependencies now fail loud: a handler whose required extra is
  not installed raises `MissingDependencyError` (carrying the package and extra
  name) instead of silently demoting to the binary fallback and producing
  fingerprints that are incomparable to a properly-installed install.
- SQLite `search()` no longer holds a write lock while reading: the offset
  histogram is aggregated in a way that does not keep a write transaction open,
  and the backend supports use as a context manager (`__enter__`/`__exit__`).
- Postgres operations roll back on error to avoid leaving an aborted
  transaction open on the shared connection.
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
- Scale-invariance design: fixed per-handler windows for sequence handlers and
  canonical 256x256 grayscale normalisation for images.
- `fingerprint-engine` CLI (`fingerprint`, `add`, `search`, `prune`) with
  configurable resolution and backend selection.
- Benchmark harness covering fingerprinting throughput, per-backend build rate
  and footprint, query latency, scaling, and accuracy at scale.
