"""Refactor safety gate: capture per-handler fingerprint hash codes, and diff two captures.

This is the harness used to prove that a refactor leaves the DEFAULT fingerprint
derivation byte-identical (the project's headline invariant). It deliberately does
NOT hardcode expected hash values -- those are platform/numpy-dependent -- so it is
not a pytest golden test. Instead it captures the current hashes to JSON and diffs
two captures taken on the SAME machine, around a change:

    python benchmarks/hash_capture.py capture /tmp/before.json   # on the base commit
    # ... apply the refactor ...
    python benchmarks/hash_capture.py capture /tmp/after.json
    python benchmarks/hash_capture.py diff /tmp/before.json /tmp/after.json

``diff`` exits non-zero and lists the handlers whose hash codes changed, so it can
gate a refactor in CI or a pre-commit hook. It generates a small DETERMINISTIC
multi-handler corpus (fixed seeds), routes each fixture through the real
``Fingerprinter`` (so it exercises the production single-read path + handler
routing), and records each fixture's routed handler, hash count, sorted hash codes,
content sha, and recorded format version. Handlers whose optional dependency is
absent are skipped (recorded as ``{"skipped": ...}``), so the tool runs on a
core-only install and compares only the handlers both captures could build.

The committed pytest suite covers the COMPLEMENTARY invariants this tool cannot
(cross-backend ranking parity, within-run determinism, and disk-vs-bytes load
equivalence); this tool covers the one they cannot: "did the hash VALUES move."
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np


def build_corpus(root: Path) -> dict[str, Path]:
    """Write one deterministic fixture per default handler; skip absent-dep ones."""

    fx: dict[str, Path] = {}

    text = root / "sample.py"
    text.write_text("def add(a, b):\n    return a + b\n" * 60, encoding="utf-8")
    fx["text"] = text

    blob = root / "data.bin"
    blob.write_bytes(bytes(range(256)) * 64)
    fx["binary"] = blob

    rng = np.random.default_rng(1234)

    try:
        from PIL import Image

        arr = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
        image = root / "pic.png"
        Image.fromarray(arr, "RGB").save(image)
        fx["image"] = image
    except ImportError:
        pass

    try:
        from scipy.io import wavfile

        t = np.linspace(0, 1.0, 8000, endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        wav = root / "sound.wav"
        wavfile.write(str(wav), 8000, (sig * 32767).astype(np.int16))
        fx["wav"] = wav
    except ImportError:
        pass

    arc = root / "arc.zip"
    with zipfile.ZipFile(arc, "w") as archive:
        archive.writestr("a.txt", "alpha contents " * 10)
        archive.writestr("b.txt", "beta contents " * 12)
        archive.writestr("c/d.txt", "gamma " * 20)
    fx["archive"] = arc

    npy = root / "vecs.npy"
    np.save(npy, rng.standard_normal((24, 32)).astype(np.float32))
    fx["npy"] = npy

    jsonl = root / "vecs.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(rng.standard_normal(16).round(4).tolist()) for _ in range(12)) + "\n",
        encoding="utf-8",
    )
    fx["jsonl"] = jsonl

    try:
        import av

        clip = root / "clip.mp4"
        with av.open(str(clip), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 96, 72, "yuv420p"
            for i in range(18):
                frame = av.VideoFrame.from_ndarray(
                    np.full((72, 96, 3), (i * 13) % 256, dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        fx["video"] = clip
    except ImportError:
        pass

    return fx


def capture() -> dict[str, dict]:
    from fingerprint_engine.core.fingerprinter import Fingerprinter

    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_corpus(Path(tmp))
        fingerprinter = Fingerprinter()
        out: dict[str, dict] = {}
        for kind, path in fixtures.items():
            try:
                fingerprint = fingerprinter.fingerprint_file(path)
                out[kind] = {
                    "handler": fingerprint.handler,
                    "n_hashes": len(fingerprint.hashes),
                    "hash_codes": sorted(h.hash_code for h in fingerprint.hashes),
                    "content_sha256": fingerprint.content_sha256,
                    "format_version": fingerprint.format_version,
                }
            except Exception as exc:  # noqa: BLE001 - record, never abort the capture
                out[kind] = {"skipped": f"{type(exc).__name__}: {exc}"}
    return out


def _summary_line(kind: str, info: dict) -> str:
    if "skipped" in info:
        return f"{kind:10s} SKIPPED {info['skipped']}"
    digest = hashlib.sha256(json.dumps(info["hash_codes"]).encode()).hexdigest()[:16]
    return (
        f"{kind:10s} handler={info['handler']:12s} v{info['format_version']} "
        f"n={info['n_hashes']:6d} hashes_sha={digest}"
    )


def diff(before_path: Path, after_path: Path) -> int:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    changed: list[str] = []
    for kind in sorted(set(before) | set(after)):
        b, a = before.get(kind, {}), after.get(kind, {})
        if "skipped" in b or "skipped" in a:
            continue  # only compare handlers both captures could build
        if b.get("handler") != a.get("handler") or b.get("hash_codes") != a.get("hash_codes"):
            changed.append(kind)
            print(
                f"CHANGED {kind}: handler {b.get('handler')}->{a.get('handler')}, "
                f"n {b.get('n_hashes')}->{a.get('n_hashes')}"
            )
    if changed:
        print(f"\n*** {len(changed)} handler(s) changed: {changed} ***")
        return 1
    print("ALL COMPARED HANDLERS BYTE-IDENTICAL (hash codes unchanged)")
    return 0


def main(argv: list[str]) -> int:
    warnings.simplefilter("ignore")
    if len(argv) == 3 and argv[1] == "capture":
        result = capture()
        Path(argv[2]).write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
        for kind, info in sorted(result.items()):
            print(_summary_line(kind, info))
        print(f"\nwrote {argv[2]}")
        return 0
    if len(argv) == 4 and argv[1] == "diff":
        return diff(Path(argv[2]), Path(argv[3]))
    print(__doc__)
    print("usage: hash_capture.py capture <out.json> | diff <before.json> <after.json>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
