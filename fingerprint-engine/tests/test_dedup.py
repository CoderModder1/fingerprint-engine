from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine import cli
from fingerprint_engine.core.dedup import DedupReport, find_duplicates
from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex


def _featured_lines(seed: int, count: int = 80) -> str:
    """Featured text so each file yields a non-empty, searchable fingerprint."""

    return "".join(
        f"def function_{seed}_{j}(value):\n    return value * {j} + {seed * 7}\n\n"
        for j in range(count)
    )


def _make_corpus(tmp_path: Path) -> dict[str, Path]:
    """A corpus with (a) an exact pair, (b) a base + lightly-edited near-dup,
    and (c) an unrelated file."""

    base_text = _featured_lines(seed=1)

    base = tmp_path / "base.py"
    base.write_text(base_text, encoding="utf-8")

    # (a) Exact byte-identical copy of base -> same content_sha256.
    exact_copy = tmp_path / "base_copy.py"
    exact_copy.write_text(base_text, encoding="utf-8")

    # (b) Lightly-edited near-duplicate of base: one body changed, one line added.
    edited = base_text.replace(
        "def function_1_5(value):\n    return value * 5 + 7\n",
        "def function_1_5(value):\n    return value * 5 + 99\n",
    ) + "# a single trailing comment line\n"
    near = tmp_path / "near.py"
    near.write_text(edited, encoding="utf-8")

    # (c) Unrelated content.
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text(
        "".join(
            f"class Widget{k}:\n    attr = {k * 13}\n    def go(self):\n        return self.attr - {k}\n\n"
            for k in range(80)
        ),
        encoding="utf-8",
    )

    return {"base": base, "exact_copy": exact_copy, "near": near, "unrelated": unrelated}


def _fingerprint_all(corpus: dict[str, Path]):
    fingerprinter = Fingerprinter()
    return [fingerprinter.fingerprint_file(path) for path in corpus.values()]


# --------------------------------------------------------------------------- #
# Library-level: the three cases resolve correctly.
# --------------------------------------------------------------------------- #


def test_exact_pair_grouped_near_pair_grouped_unrelated_singleton(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    fingerprints = _fingerprint_all(corpus)

    report = find_duplicates(fingerprints, min_confidence=0.5)
    assert isinstance(report, DedupReport)

    # (a) EXACT: base and its byte-identical copy share one content_sha256 and
    # land in exactly one exact cluster together.
    assert len(report.exact) == 1
    exact_paths = {Path(p).name for p in report.exact[0].paths}
    assert exact_paths == {"base.py", "base_copy.py"}

    # (b) NEAR: base (the distinct content of the exact pair) and the edited
    # near-dup are clustered, with confidence at or above the threshold.
    assert len(report.near) == 1
    near_cluster = report.near[0]
    near_names = {Path(p).name for p in near_cluster.paths}
    assert near_names == {"base.py", "near.py"}
    assert near_cluster.confidence >= 0.5

    # (c) The unrelated file is in no cluster: distinct contents are
    # base, near, unrelated -> base+near clustered, unrelated alone.
    assert report.singletons == 1
    assert report.total_paths == 4
    assert report.total_distinct == 3


def test_exact_tier_short_circuits_identical_paths(tmp_path: Path) -> None:
    """Identical paths appear only in the exact cluster, not re-compared fuzzily."""

    corpus = _make_corpus(tmp_path)
    fingerprints = _fingerprint_all(corpus)
    report = find_duplicates(fingerprints, min_confidence=0.5)

    # The near cluster reports ONE path per distinct content (the representative),
    # so the byte-identical copy never shows up a second time in the near tier.
    near_paths = [p for cluster in report.near for p in cluster.paths]
    assert near_paths.count(str(corpus["base"].resolve())) <= 1
    assert str(corpus["exact_copy"].resolve()) not in near_paths


def test_high_threshold_drops_near_duplicate(tmp_path: Path) -> None:
    """A near-1.0 cutoff keeps the exact pair but rejects the fuzzy match."""

    corpus = _make_corpus(tmp_path)
    fingerprints = _fingerprint_all(corpus)

    report = find_duplicates(fingerprints, min_confidence=0.999)

    # Exact (byte-identical) detection is independent of the fuzzy threshold.
    assert len(report.exact) == 1
    # The edited file no longer clears the bar, so no near cluster forms. The
    # base content is still clustered (in the exact tier), so only the near and
    # unrelated distinct contents are singletons -- a distinct content in ANY
    # cluster is not counted as a singleton.
    assert report.near == []
    assert report.singletons == 2


def test_all_distinct_no_duplicates(tmp_path: Path) -> None:
    base = tmp_path / "a.py"
    base.write_text(_featured_lines(seed=1), encoding="utf-8")
    other = tmp_path / "b.py"
    other.write_text(
        "".join(
            f"class Thing{k}:\n    v = {k * 11}\n    def run(self):\n        return self.v + {k}\n\n"
            for k in range(80)
        ),
        encoding="utf-8",
    )
    fingerprinter = Fingerprinter()
    fingerprints = [fingerprinter.fingerprint_file(base), fingerprinter.fingerprint_file(other)]

    report = find_duplicates(fingerprints, min_confidence=0.5)

    assert report.exact == []
    assert report.near == []
    assert report.singletons == 2
    assert report.total_distinct == 2


def test_empty_input() -> None:
    report = find_duplicates([], min_confidence=0.5)
    assert report.exact == []
    assert report.near == []
    assert report.singletons == 0
    assert report.total_paths == 0
    assert report.total_distinct == 0


def test_find_duplicates_accepts_caller_index(tmp_path: Path) -> None:
    """A caller-supplied index backend is used as scratch space and gives the
    same clustering as the default in-memory one."""

    corpus = _make_corpus(tmp_path)
    fingerprints = _fingerprint_all(corpus)

    supplied = InMemoryHashIndex()
    report = find_duplicates(fingerprints, min_confidence=0.5, index=supplied)

    assert len(report.exact) == 1
    assert len(report.near) == 1
    # The representatives were added to the supplied index (one per distinct
    # content): base, near, unrelated.
    assert supplied.file_count == 3


# --------------------------------------------------------------------------- #
# CLI-level: dedup subcommand emits the JSON report.
# --------------------------------------------------------------------------- #


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_dedup_emits_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = _make_corpus(tmp_path)
    argv = [
        "dedup",
        "--min-confidence",
        "0.5",
        *[str(path) for path in corpus.values()],
    ]

    code, out, _err = _run(argv, capsys)

    assert code == 0
    payload = json.loads(out)
    assert payload["min_confidence"] == 0.5
    assert payload["skipped"] == []
    assert payload["total_paths"] == 4
    assert payload["total_distinct"] == 3

    # Exact cluster groups the byte-identical pair.
    assert payload["exact_cluster_count"] == 1
    exact_names = {Path(p).name for p in payload["exact_clusters"][0]["paths"]}
    assert exact_names == {"base.py", "base_copy.py"}

    # Near-duplicate cluster groups the edited pair with its confidence.
    assert payload["near_duplicate_cluster_count"] == 1
    near = payload["near_duplicate_clusters"][0]
    assert {Path(p).name for p in near["paths"]} == {"base.py", "near.py"}
    assert near["confidence"] >= 0.5

    # The unrelated file is the lone singleton.
    assert payload["singletons"] == 1


def test_cli_dedup_fail_soft_reports_bad_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _make_corpus(tmp_path)
    missing = tmp_path / "nope.py"
    argv = [
        "dedup",
        str(corpus["base"]),
        str(missing),
        str(corpus["exact_copy"]),
    ]

    code, out, _err = _run(argv, capsys)

    assert code == 0  # one bad path does not abort the batch
    payload = json.loads(out)
    skipped_paths = [item["path"] for item in payload["skipped"]]
    assert str(missing) in skipped_paths
    reason = next(item["reason"] for item in payload["skipped"] if item["path"] == str(missing))
    assert reason.startswith("FileNotFoundError")
    # The two good (identical) paths still form an exact cluster.
    assert payload["exact_cluster_count"] == 1
