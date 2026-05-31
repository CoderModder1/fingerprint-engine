# PR: Hardening — Tier-1 correctness/DoS, Tier-2 observability + Apache-2.0, Tier-3 routing

> Scratch file (untracked, not committed). Branch: `hardening-tier1-2` (2 commits, based on `main`).
> Delete this file once the PR is open.

---

## Title

```
Engine hardening (Tier 1-3) + performance (Tier 4)
```

## Body

Hardening + performance pass from a multi-agent audit of the engine. Three commits; `main` is unchanged.

**Tier 1 — correctness & data-loss**
- Fail-loud on a missing optional dependency (Pillow/scipy/pydub/pypdf) instead of silently demoting to incomparable binary fingerprints — new `core/exceptions.py` hierarchy.
- SQLite `search()` no longer leaves an open write transaction / write lock (+ context managers on the SQL backends).
- Atomic, durable snapshot writes (temp + fsync + `.bak` + `os.replace`) with `.bak` recovery on a corrupt primary.
- Fail-soft, order-preserving batch fingerprinting (one bad file no longer aborts the batch).
- Constant/degenerate-signal false-match suppression (distinct constants no longer collide).
- Adaptive-window cross-length fix: fixed-window handlers use an authoritative window so different-length copies of the same content match.
- `max_file_size_bytes` guard rejecting oversized files *before* the read (`FileTooLargeError`) — closes an unbounded-read OOM/DoS vector.

**Tier 2 — hardening, observability, packaging**
- Full `FingerprintConfig.validate` (bounds `peak_percentile`/`peak_threshold`).
- Snapshot `schema_version` (absent = legacy, unknown = `InvalidSnapshotError`) + value validation.
- Concurrency contract: in-memory writes serialized by an `RLock`; SQLite WAL + `busy_timeout`.
- Structured logging (`getLogger(__name__)`, `NullHandler` on the package root).
- CLI clean error messages with distinct exit codes (0/1/2/3/4) and a structured `skipped[]` list.
- **Apache-2.0** LICENSE + NOTICE, PEP 639 metadata + classifiers, `__all__` + `__version__`; `requirements.txt` removed in favor of the `[dev]` extra; `CHANGELOG.md`.
- `max_pdf_pages` cap wired end-to-end via a new `FileHandler.configure(config)` hook; `SECURITY.md` documenting the decoder attack surface.

**Tier 3 — routing & docs**
- `can_handle` returns the MAX of extension/MIME/prefix scores (a strong MIME signal is no longer masked by a weaker extension match).
- Audio no longer force-decodes non-MP3 (`.ogg`/`.flac`/`.m4a`/…) as `format="mp3"`.
- Text sniffer scores printability on decoded code points (`char.isprintable`) so accent-dense latin-1 text is accepted instead of demoted to binary.
- SVG / text-based image MIME types excluded from the image handler.
- PDF structural-only (latin-1) fallback now warns it is not real text extraction.
- Cross-backend parity tests (offset tie-break, ranking, snapshot interop).
- Benchmark refreshed on current HEAD (`RESULTS.md` post-optimization; SQLite 1000-file query mean **6633 ms → 1252 ms**, ~5.3×); pre-optimization data preserved in `pre-optimization-results.json`.

**Tier 4 — performance (all output-preserving; fingerprint hashes + search ranks byte-identical, verified against the prior commit)**
- Vectorized `extract_peaks` (separable `-inf`-padded numpy 3×3 local max) — fingerprint throughput ~30 → ~44 files/s (≈3× on small/dense inputs).
- Opt-in process-pool batch fingerprinting (`fingerprint_many(executor="process")`); thread mode stays the default.
- Bulk transactional ingest `HashIndex.add_many()` (SQLite one-transaction + `synchronous=NORMAL`, Redis pipeline, Postgres `COPY`) — SQLite ingest 3.0–4.4× faster, proven identical to sequential `add()`.
- Slotted `IndexPosting`/`LandmarkPoint`/`ConstellationHash` (memory at scale).
- Benchmark refresh + fanout/max-peaks sweep (~60% fewer hashes/file at recall@1 = 1.0).

**Deferred (called out, not silently dropped)**
- A true per-decode media *timeout* (`signal.alarm` can't run in the `ThreadPoolExecutor` workers); the `max_file_size_bytes` + `max_pdf_pages` caps bound the worst cases instead.
- Tier-4 follow-ups: integer `file_id` surrogate key + compact posting encoding, SQL `top_k`/HAVING fan-out pushdown, ANN/LSH candidate generation, streaming ingest.

**Tests:** 154 passed / 12 skipped (skips are the optional-backend / live-Postgres suites); ruff + mypy clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## How to push + open the PR (when you have a GitHub remote)

```bash
cd /Users/auto/Desktop/Claude

# 1. point at your GitHub repo (replace with your URL)
git remote add origin git@github.com:<you>/<repo>.git   # or https://github.com/<you>/<repo>.git

# 2. (first push only) publish main so the PR has a base
git push -u origin main

# 3. push the feature branch
git push -u origin hardening-tier1-2

# 4a. open the PR with the GitHub CLI (after: brew install gh && gh auth login)
gh pr create --base main --head hardening-tier1-2 \
  --title "Engine hardening (Tier 1-3) + performance (Tier 4)" \
  --body-file PR_DESCRIPTION.md

# 4b. ...or just open the compare page in a browser
#     https://github.com/<you>/<repo>/compare/main...hardening-tier1-2?expand=1
```
