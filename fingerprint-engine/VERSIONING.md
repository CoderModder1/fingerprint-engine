# Versioning and Compatibility

This document defines what `fingerprint-engine` promises to keep stable, how
those promises are versioned, and the checklist that must be true to tag the
first stable release (`1.0.0`). Until then the project is pre-1.0 and the public
surface may still change; this document is the contract we intend to freeze at
1.0.

## Semantic versioning policy

The project follows [Semantic Versioning 2.0.0](https://semver.org). Given a
version `MAJOR.MINOR.PATCH`:

- **MAJOR** -- incremented for a backward-incompatible change to any *stable
  public interface* enumerated below (a removed/renamed public symbol, a changed
  method signature or return shape, a snapshot `schema_version` the new build can
  no longer load, or a change to the *default* fingerprint derivation / search
  ranking).
- **MINOR** -- incremented for backward-compatible additions: a new handler,
  backend, CLI subcommand, optional extra, keyword-only argument with a default,
  or an **opt-in, default-off** matching/scoring feature that leaves the default
  behavior byte-identical.
- **PATCH** -- incremented for backward-compatible bug fixes that do not change
  any documented interface, the default fingerprint derivation, or the default
  ranking.

Pre-1.0 (`0.y.z`), the same intent applies on a best-effort basis but MINOR
bumps may carry breaking changes; the guarantees below become binding at 1.0.

### What is *not* covered by these guarantees

Anything prefixed with `_`, the `benchmarks/` harness, the test suite, exact log
message text, exact floating-point `score` *values* (only the ordering and the
`confidence` semantics are stable -- see below), and internal backend storage
layouts (the SQLite table DDL, the Redis key layout) that are not the portable
JSON snapshot. Optional extras' transitive dependency versions float within the
declared lower bounds in `pyproject.toml`.

## Stable public interfaces

The public API is what `import fingerprint_engine` re-exports (its `__all__`),
the `fingerprint-engine` CLI commands and their JSON output keys, and the
portable index snapshot format. The following interfaces carry compatibility
guarantees.

### 1. Snapshot JSON schema (`schema_version`, currently `1`)

`InMemoryHashIndex.save()` writes, and `.load()`/`from_dict()` read, a portable
JSON snapshot with this top-level shape:

```json
{
  "schema_version": 1,
  "backend": "in_memory",
  "files": { "<file_id>": [[<hash_code>, <time_offset>], ...] },
  "metadata": { "<file_id>": { "path": "...", "handler": "...", "hash_count": N, ... } }
}
```

Guarantees:

- The current build reads every snapshot whose `schema_version` is in its
  supported set (`{1}` today). An **absent** `schema_version` is treated as
  version 1 (legacy snapshots written before the field existed remain loadable).
- A snapshot whose `schema_version` is present but unsupported, or whose
  structure is invalid, is rejected loudly with `InvalidSnapshotError` (a
  subclass of both `FingerprintError` and `ValueError`) -- never loaded as
  partial or misinterpreted state.
- `schema_version` is bumped only on a backward-incompatible change to the
  snapshot layout, and a `schema_version` bump is a **MAJOR** release. A build
  may add support for reading *new* versions in a MINOR release as long as it
  still reads version 1.

The schema version concerns the *container* (how postings are serialized), and
is independent of the fingerprint derivation version discussed in section 4.

### 2. The `HashIndex` contract

`HashIndex` is the storage-agnostic abstract base shared by all four backends
(`InMemoryHashIndex`, `SQLiteHashIndex`, `RedisHashIndex`, `PostgresHashIndex`).
The following members are part of the stable contract; every backend implements
them with identical observable semantics:

- `file_count` / `posting_count` -- counts.
- `add(fingerprint)` / `add_many(fingerprints)` -- ingest one or many
  fingerprints; `add_many` is the bulk/transactional form and is observably
  equivalent to `add()` in sequence.
- `remove(file_id)` -- delete a file's postings and metadata.
- `query(hash_code)` / `query_many(hash_codes)` -- raw posting lookup, batched.
- `search(fingerprint, top_k=..., calibration=...)` -- ranked matches (see
  section 3).
- `list_files()` -- the indexed `file_id`s in sorted order.
- `iter_metadata()` -- per-file metadata dicts, yielded in the same
  backend-independent sorted `file_id` order.
- `contains(file_id)` / `file_id in index` -- membership, used by incremental
  ingest to skip already-indexed content cheaply.
- `prune_stop_hashes(max_df_ratio=...)` -- explicit, caller-invoked pruning of
  non-discriminative high document-frequency codes (does **not** run by
  default).
- `save(path)` / `load(path)` -- the portable snapshot (section 1).

Note: `list_files`, `iter_metadata`, `contains`, `add_many`, and `query_many`
are part of the contract **now** -- adding them was a backward-compatible
addition, and they will not be removed without a MAJOR bump.

Guarantee: a method's signature and observable behavior, and the set of contract
methods, are stable across MINOR/PATCH. New backends or new optional keyword
arguments (with defaults preserving current behavior) may be added in a MINOR
release.

### 3. `SearchResult` and confidence scoring semantics

`search()` returns a list of frozen `SearchResult` records, each with:
`file_id`, `score`, `confidence`, `aligned_votes`, `total_votes`,
`unique_hashes`, `offset`, and `metadata`. Stable semantics:

- **`confidence`** is a handler-independent value in `[0, 1]`: the fraction of
  the *smaller* fingerprint's hashes that aligned at the winning time offset
  (`aligned_votes / min(query_hash_count, target_hash_count)`, capped at 1.0).
  It is comparable across content types, so a single `Calibration`
  `default_min_confidence` is meaningful, with optional `per_handler` overrides.
- **Ranking** is by descending `score`, with deterministic tie-breaks
  (descending `aligned_votes`, then descending `unique_hashes`, then ascending
  `file_id`). The winning offset within a file is the histogram bin with the
  most votes, ties broken by the smallest offset.
- **`score`** is the raw ranking value. Its tie-break ordering and the meaning
  of the component fields are stable; the exact floating-point `score` *value*
  and the relative weights of its terms are an implementation detail and may be
  tuned within a MINOR release (ordering of clearly-distinct matches is what is
  promised, not the literal number).
- All backends produce identical `SearchResult`s for the same index contents and
  query, because aggregation feeds one shared scoring/ranking path.

A change to the **default** `confidence` formula or to the ranking tie-break
rules is a **MAJOR** change. New scoring behavior is shipped opt-in and
default-off (e.g. via `Calibration` or a future config flag) so the default
ranking stays byte-identical.

### 4. Fingerprint derivation

Hashes are deterministic given **the same `FingerprintConfig` and the same input
format/handler**. Two installs that fingerprint the same bytes with the same
config and the same handler version produce byte-identical hash codes and time
offsets; this is what makes an index portable and a self-match exact.

Caveats and guarantees:

- Hashes are stable only *within* the same `FingerprintConfig`. Changing tuning
  parameters (`window_size`, `hop_size`, `peak_*`, `constellation_fanout`,
  `hash_bits`, the per-handler fixed windows, ...) changes the hashes by design,
  so an index built with one config must be searched with the same config.
- The **default** derivation -- the hashes produced under the default
  `FingerprintConfig` and the shipped handlers -- is a stable interface. Changing
  it requires re-indexing existing corpora, so it is gated behind a
  **`fingerprint_format_version` bump** and is a **MAJOR** release.
- Most opt-in matching features (for example hash quantization, or a *global*
  `window_bank` applied to every handler) are **default-off**: when enabled they
  change the derived hashes (and bump `effective_format_version` via a per-flag
  offset) without altering the default-config hashes for callers who leave them
  off. **Exception (v2):** the *audio* handler now applies a multi-resolution
  window bank **by default** — a deliberate default-derivation change that was
  promoted via the v2 `FINGERPRINT_FORMAT_VERSION` bump and a re-index, exactly
  per the enforcement rule below. A core-only install (numpy only) and an install
  with optional handler extras produce the same hashes for the content types they
  share.

#### 4a. `FINGERPRINT_FORMAT_VERSION` — the enforced derivation version

The fingerprint derivation now carries an explicit, machine-checked version,
implemented as the module constant `FINGERPRINT_FORMAT_VERSION` (currently `2`)
in `fingerprint_engine/core/models.py`.

> **v2 (2026-05-31).** The default derivation changed, so a v1 corpus must be
> re-indexed (the version check below detects a v1 query against a v2 index).
> Two changes: (1) the signal/spectrogram reductions (`mean`/`std`/`percentile`)
> now accumulate in **float64** for cross-platform reproducibility — output is
> identical for the non-audio handlers on real inputs, but it corrects a
> near-zero-mean signal's normalisation, so **audio** hash codes change; and
> (2) the **audio** handler now fingerprints with a multi-resolution **window
> bank by default** (`AudioFileHandler.default_window_bank`), so audio
> excerpt/clip matching works out of the box at ~N× the audio postings. The
> seven non-audio handlers are output-identical to v1; only the version *stamp*
> advances (one version is pinned per index, so the bump is global). It is **distinct from the snapshot
`schema_version`** (section 1): `schema_version` versions the JSON *container*
that serializes postings, whereas `FINGERPRINT_FORMAT_VERSION` versions the
*meaning of the `hash_code` integers* inside it. Two builds can share a snapshot
schema yet derive incompatible hash codes; only an equal format version
guarantees the codes occupy the same code space.

How it travels and is enforced (all default-preserving — it adds metadata, never
a hash code or a ranking):

- **Stamped onto each fingerprint.** `Fingerprinter` records the *effective*
  format version under the `fingerprint_format_version` key of
  `Fingerprint.config` (additive; it does not displace any tuning key), readable
  as `Fingerprint.format_version`. The value is computed by
  `effective_format_version(config)`: a default config (and any config whose
  hash-changing fields are all default) reports the bare
  `FINGERPRINT_FORMAT_VERSION`, so existing fingerprints/indexes are byte-for-byte
  unchanged.
- **Stamped into the index and snapshot.** A `HashIndex` pins its
  `format_version` from the first fingerprint added; `save()` writes it as the
  top-level snapshot field `fingerprint_format_version` (alongside
  `schema_version`), and `load()`/`from_dict()`/`load_snapshot()` restore it. An
  **absent** field loads as the default (legacy snapshots stay loadable and
  compatible).
- **Detected at search and at add.** `HashIndex.search()` compares the query
  fingerprint's `format_version` with the index's; a mismatch emits a
  `RuntimeWarning` by default (so no existing pipeline breaks) and raises
  `FormatVersionMismatchError` when called with `strict_format=True`. Adding a
  fingerprint whose version differs from an already-pinned index warns and keeps
  the index's pinned version (first writer wins). A *matching*-version query is a
  no-op: rankings are byte-identical to before the check existed.
- **Opt-in hash-changers bump the recorded version.** Enabling
  `freq_quantization > 1`, a `window_bank`, or `image_mode == "phash"` makes
  `effective_format_version` report a *distinct* value (each flag a distinct,
  composable offset). An index built with such a flag is therefore detectably
  incompatible with a default index — without flipping any default.

**Enforcement rule.** Flipping ANY hash-changing default — promoting an opt-in
flag to default-on, or changing the constellation packing / per-handler windows /
canonical image transform — **REQUIRES bumping `FINGERPRINT_FORMAT_VERSION` and
re-indexing** existing corpora, and is a **MAJOR** release. The mechanism above
now *enforces detection* of the resulting incompatibility: a query or snapshot at
the old version is flagged against an index at the new version rather than
silently returning false matches. (Promoting an opt-in flag does not change the
value `effective_format_version` already reports for that flag; it changes which
config is the *default*, so the new default's recorded version differs from the
old default's, which is exactly the incompatibility the check surfaces.)

The set of hash-changing config fields is declared explicitly as
`models.HASH_CHANGING_FIELDS` (kept in lock-step with `effective_format_version`
by a test), so "which knobs change the derivation" is greppable rather than
implicit in the function body.

#### 4b. Handler-local derivation changes — the over-bump, the escape hatch, and the target model

`FINGERPRINT_FORMAT_VERSION` is a **single scalar pinned per index**, but an
index may hold a *mix* of handlers (text + image + audio + …). So a derivation
change confined to ONE handler still forces a **global** bump and a re-index of
the *unchanged* handlers' corpora. The v2 release is exactly this case: only the
audio derivation changed (float64 reductions + the default window bank), yet the
seven non-audio handlers — byte-identical in output — were re-stamped v2 and a v1
corpus of them must be rebuilt. The version's granularity (whole-corpus) does not
match compatibility's true granularity (per-handler).

**Escape hatch (adopted, the cheap option).** A *handler-local* derivation change
need not be shipped as a global default bump. It can instead ship as a new
**opt-in flag with its own `_FORMAT_BUMP_*` offset** (the mechanism already in
`effective_format_version`): callers who leave it off keep matching at the
baseline, and only opt-in users get the new version — no forced re-index of
unchanged corpora. The v2 audio bank was promoted to default-on *by deliberate
choice* (it makes the "Shazam-style" excerpt-matching claim honest out of the
box); a future handler-local change MAY instead stay opt-in until a coordinated
major bump. Prefer this for any change that touches a single handler.

**Target model (designed, not yet built): a composite per-handler version.**
The structurally correct fix is to make the recorded version a small mapping of
*handler-family → derivation version* (e.g. `{"audio": 2, "default": 1}`) instead
of one scalar, so a per-handler bump leaves the other families untouched. Sketch:

- `effective_format_version` returns the composite for a config; `Fingerprint`
  carries its handler's entry; the index pins the *union* of the families it
  holds rather than one number.
- `_check_format_version` resolves compatibility **per handler family**: a query
  is compatible iff the index's version for *that family* matches. An audio-only
  v→v+1 bump then flags only audio queries/corpora; text/image stay compatible.
- **Migration:** a legacy scalar `v` maps to the uniform composite `{"*": v}`
  (and an absent stamp still reads as the current default), so existing snapshots
  load unchanged under the tolerant "absent ⇒ default" rule.

This is deliberately deferred (it is a snapshot-schema-level change — see §1 — and
must preserve the cross-backend ranking parity invariant), and is the recommended
direction *if/when* mixed-handler corpora with independently-evolving handlers
become a real workload. The heavier alternative — a per-file or per-posting
version tag allowing side-by-side same-handler versions in one index — is
**explicitly out of scope** unless that capability is actually required, because
it is the only option that touches every backend's storage layout and the shared
scoring path (the two riskiest surfaces in the codebase).

## 1.0 definition-of-done checklist

`1.0.0` is tagged when **all** of the following are true:

- [ ] **Public API frozen.** `fingerprint_engine.__all__`, the four `HashIndex`
      backends and the full contract in section 2, and the model dataclasses
      (`FingerprintConfig`, `Fingerprint`, `SearchResult`, `Calibration`,
      `LandmarkPoint`, `ConstellationHash`) are reviewed and declared stable.
- [ ] **Snapshot schema documented and versioned.** `schema_version` is `1`,
      load validates it, and unsupported/invalid snapshots fail with
      `InvalidSnapshotError`. (Met today.)
- [ ] **Fingerprint derivation pinned.** The default-config derivation is
      documented as the stable interface, an explicit `fingerprint_format_version`
      is recorded, and every opt-in matching feature is verified default-off and
      additive (default-config hashes byte-identical with the feature compiled
      in but not enabled).
- [ ] **CLI contract stable.** The `fingerprint`, `add`, `search`, `prune`,
      `list`, `dedup`, and `doctor` subcommands, their flags, JSON output keys,
      and exit codes (0 success; 1 input error; 2 usage; 3 missing dependency;
      4 backend/operational) are documented and covered by tests.
- [ ] **Dependency boundary verified.** numpy is the only hard runtime
      dependency; every other capability is behind an extra
      (`image`/`audio`/`pdf`/`redis`/`postgres`/`service`), lazily imported, and
      a core-only install imports and runs the core handlers/backends. `doctor`
      reports the true availability.
- [ ] **Quality gates green.** `pytest`, `ruff check .`, and
      `mypy fingerprint_engine` all pass on every supported Python
      (3.10--3.13) in CI.
- [ ] **Docs complete.** `README`, `CHANGELOG` (an Unreleased section promoted to
      the `1.0.0` heading), `SECURITY.md`, `LICENSE`/`NOTICE`, and this
      `VERSIONING.md` are current and consistent.
- [ ] **Accuracy baseline recorded.** The deterministic accuracy harness
      (`benchmarks/accuracy.py` + `tests/test_accuracy.py`) records the shipped
      recall/precision baseline. As of v2, audio excerpt/clip recall is part of
      the *default* baseline (the float64 reductions + the default audio window
      bank fixed the former limitation), so a regression in it is detectable.
