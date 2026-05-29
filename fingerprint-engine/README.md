# Universal File Fingerprinting Engine

A modular file fingerprinting engine inspired by Shazam's landmark-based audio matching. Each file type is converted into a 1D signal, transformed into a spectrogram-like matrix, reduced to landmark peaks, paired into constellations, and indexed as compact searchable hashes.

## Architecture

The engine has three layers:

1. Core orchestration
   - `core/fingerprinter.py` discovers `FileHandler` plugins from the `handlers` package.
   - `core/models.py` defines `Fingerprint`, `LandmarkPoint`, `ConstellationHash`, and tuning config dataclasses.
   - `core/index.py` defines the storage-agnostic `HashIndex` contract plus the default dict-backed index.

2. FFT-equivalent pipeline
   - `core/fft_pipeline.py` normalizes handler signals, applies sliding windows, runs `numpy.fft.rfft`, extracts adaptive local maxima, builds peak-pair constellations, and hashes `(freq1, freq2, delta_t)` into deterministic integer codes.
   - Binary files use normalized raw bytes.
   - Text and source files use character code, character class, and token-length rhythm.
   - Images use flattened grayscale pixel intensity.
   - Audio uses decoded mono samples for WAV, and MP3 via `pydub`/ffmpeg when available.
   - PDFs use extracted page text plus page boundary markers.

3. Scalability boundaries
   - Handlers are auto-discovered plugins, not selected by hardcoded type switches in the core.
   - Fingerprint resolution is configurable from Python or the CLI.
   - Batch fingerprinting uses `ThreadPoolExecutor`.
   - `HashIndex` can be replaced with Redis, SQLite, Postgres, or another backend without changing handlers or the pipeline.

## Install

```bash
cd /Users/auto/Desktop/Claude/fingerprint-engine
python -m pip install -r requirements.txt
```

MP3 support requires `pydub` and a working ffmpeg installation.

## CLI Usage

Fingerprint a file:

```bash
python cli.py fingerprint path/to/file
```

Add files to the default local JSON index:

```bash
python cli.py add path/to/file1 path/to/file2
```

Search the index with a query file:

```bash
python cli.py search path/to/query --top-k 5
```

Use a custom index path:

```bash
python cli.py --index-path ./my-index.json add path/to/file
python cli.py --index-path ./my-index.json search path/to/query
```

Tune resolution:

```bash
python cli.py --window-size 2048 --hop-size 512 --fanout 8 fingerprint path/to/file
```

## Python Usage

```python
from core.fingerprinter import Fingerprinter
from core.index import InMemoryHashIndex
from core.models import FingerprintConfig

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

## Extension Guide

### Add A File Handler

Create a module in `handlers/` containing a subclass of `FileHandler`:

```python
from pathlib import Path
import numpy as np
from handlers.base import FileHandler

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

### Add An Index Backend

Subclass `HashIndex` in `core/index.py` or a new module and implement:

- `add(fingerprint)`
- `remove(file_id)`
- `query(hash_code)`
- `search(fingerprint, top_k)`
- `save(path)`

Keep `query(hash_code)` returning postings with `file_id`, `hash_code`, and `time_offset`; the time-alignment scoring can then be reused or pushed into the backend.

## Tests

```bash
cd /Users/auto/Desktop/Claude/fingerprint-engine
pytest
```
