# Security

This document describes the untrusted-input posture of `fingerprint_engine` and
the controls the library provides. It is intended for anyone exposing the
engine to files they did not produce (e.g. user uploads).

## Threat model

Fingerprinting reads a file's bytes and routes them through a content handler.
Several handlers hand those bytes to third-party media decoders that parse
complex, attacker-controllable formats:

- **PDF** (`pdf` extra, `pypdf`) — `pdf_handler.py` parses untrusted PDF bytes
  with `pypdf` in-process.
- **MP3 audio** (`audio` extra, `pydub`) — `audio_handler.py` calls
  `pydub.AudioSegment.from_file(..., format="mp3")`, which shells out to an
  **`ffmpeg` subprocess** to decode the audio.
- **Images** (`image` extra, `Pillow`) — `image_handler.py` decodes untrusted
  image bytes with `Pillow`/`PIL`.
- **WAV audio** — decoded with `scipy.io.wavfile` in-process.

These decoders are large native/third-party codebases. A malformed or hostile
file can trigger excessive memory/CPU use or, in the worst case, a decoder
vulnerability. The engine does **not** sandbox them for you.

## Controls the library provides

### Resource limits (`FingerprintConfig`)

Two finite, validated knobs bound the most direct denial-of-service vectors.
Both must be `>= 0`; `0` means *unlimited* (explicit opt-out). They are
validated by `FingerprintConfig.validate()`.

- **`max_file_size_bytes`** (default `256 * 1024 * 1024`, i.e. 256 MiB) — bounds
  the OOM vector from a huge input. `Fingerprinter.fingerprint_file` stats the
  file and raises `FileTooLargeError` **before** `read_bytes()` loads the whole
  file into memory, so an oversized file is never read. The default sits far
  above any normal source/image/PDF/audio file. CLI flag: `--max-file-size`.
- **`max_pdf_pages`** (default `0` = unlimited) — caps how many PDF pages the
  PDF handler will decode, bounding a "page bomb". Enforcement lives in the PDF
  handler; this is the configuration knob. CLI flag: `--max-pdf-pages`.

`FileTooLargeError` is a `FingerprintError`. In batch mode
(`Fingerprinter.fingerprint_many`, fail-soft by default) an oversized file is
skipped like any other per-file failure, so one bad file never aborts the batch.
On the single-file CLI path it maps to exit code `1` (an input error).

### Snapshot load trust boundary

The in-memory index can persist and reload a JSON snapshot. Treat a snapshot as
trusted only if you control its source. The loader validates structure before
trusting it:

- The top-level `schema_version` is checked against the supported set; an
  unsupported version is rejected with `InvalidSnapshotError` rather than being
  silently misinterpreted. (An absent version is treated as the legacy v1.)
- Structural shape and field values are validated on load (e.g. `files` must be
  a mapping); malformed or corrupt primary/backup snapshots raise
  `InvalidSnapshotError`.

The snapshot format is plain JSON — it is **not** executed and does **not** use
pickle, so loading a snapshot cannot run arbitrary code. The validation above
guards against malformed/corrupt data, not against a maliciously crafted but
structurally valid snapshot you chose to trust.

## Recommendations for untrusted uploads

Do not expose fingerprinting of untrusted uploads without OS-level isolation in
addition to the library knobs:

1. Set `max_file_size_bytes` (and `max_pdf_pages` for PDFs) to the smallest
   values your workload tolerates.
2. Run the decoders sandboxed and resource-limited at the OS level — e.g. a
   container/VM with CPU, memory, wall-clock (`ulimit`/cgroups) and no outbound
   network, and an isolated, throwaway filesystem. The in-process knobs cannot
   contain a decoder that hangs or crashes natively.
3. Keep optional decoders (`pypdf`, `Pillow`, `pydub` + `ffmpeg`, `scipy`)
   patched; subscribe to their security advisories.
4. Only load index snapshots from sources you control.

## Reporting

Report suspected vulnerabilities privately to the maintainer rather than via a
public issue.
