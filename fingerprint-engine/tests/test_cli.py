from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine import cli
from fingerprint_engine.core.exceptions import MissingDependencyError

# Backends driven end-to-end through main(). Redis/Postgres need a live server,
# so the file-backed memory store and the embedded SQLite store are exercised.
BACKENDS = ["memory", "sqlite"]


def _write_corpus(tmp_path: Path) -> list[Path]:
    """A few featured text files so each yields a searchable (non-empty) fingerprint."""

    paths: list[Path] = []
    for i in range(3):
        path = tmp_path / f"doc_{i}.py"
        lines = [f"def function_{i}_{j}(value):\n    return value * {j} + {i * 7}\n\n" for j in range(60)]
        path.write_text("".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def _backend_argv(backend: str, tmp_path: Path) -> list[str]:
    """Global flags selecting the backend and its on-disk location under tmp_path."""

    if backend == "sqlite":
        return ["--backend", "sqlite", "--sqlite-path", str(tmp_path / "index.sqlite3")]
    return ["--backend", "memory", "--index-path", str(tmp_path / "index.json")]


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("backend", BACKENDS)
def test_fingerprint_command_emits_json(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_corpus(tmp_path)[0]

    code, out, _err = _run(_backend_argv(backend, tmp_path) + ["fingerprint", str(target)], capsys)

    assert code == 0
    payload = json.loads(out)
    assert payload["handler"] == "text"
    assert payload["hash_count"] > 0
    assert Path(payload["path"]).name == target.name


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_skips_bad_path_and_indexes_good(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _write_corpus(tmp_path)
    missing = tmp_path / "nope.py"
    files = [str(good[0]), str(missing), str(good[1]), str(good[2])]

    base = _backend_argv(backend, tmp_path)
    code, out, _err = _run(base + ["add", *files], capsys)

    assert code == 0
    payload = json.loads(out)
    # The bad path is reported structurally, naming the offending path and reason.
    skipped_paths = [item["path"] for item in payload["skipped"]]
    assert str(missing) in skipped_paths
    skipped_reason = next(item["reason"] for item in payload["skipped"] if item["path"] == str(missing))
    assert skipped_reason.startswith("FileNotFoundError")
    # The three good files are indexed despite the bad one in the middle.
    indexed_names = {Path(item["path"]).name for item in payload["indexed_files"]}
    assert indexed_names == {"doc_0.py", "doc_1.py", "doc_2.py"}
    assert payload["file_count"] == 3

    # Re-open the store and confirm the additions persisted (saved).
    code2, out2, _err2 = _run(base + ["add", str(good[0])], capsys)
    assert code2 == 0
    assert json.loads(out2)["file_count"] == 3  # already present; count is stable


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_self_match(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _write_corpus(tmp_path)
    base = _backend_argv(backend, tmp_path)

    add_code, add_out, _ = _run(base + ["add", *[str(p) for p in good]], capsys)
    assert add_code == 0
    query_id = next(
        item["file_id"]
        for item in json.loads(add_out)["indexed_files"]
        if Path(item["path"]).name == good[0].name
    )

    code, out, _err = _run(base + ["search", str(good[0])], capsys)

    assert code == 0
    payload = json.loads(out)
    result_ids = [result["file_id"] for result in payload["results"]]
    assert query_id in result_ids, "a file must self-match when searched against the index"


@pytest.mark.parametrize("backend", BACKENDS)
def test_prune_command(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _write_corpus(tmp_path)
    base = _backend_argv(backend, tmp_path)
    assert _run(base + ["add", *[str(p) for p in good]], capsys)[0] == 0

    code, out, _err = _run(base + ["prune", "--max-df-ratio", "0.5"], capsys)

    assert code == 0
    payload = json.loads(out)
    assert payload["backend"] == backend
    assert payload["pruned_postings"] >= 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_dependency_exits_3(
    backend: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _write_corpus(tmp_path)[0]
    base = _backend_argv(backend, tmp_path)

    # Force the routed handler's load() to raise MissingDependencyError: the CLI
    # must surface a clean one-line error and exit 3, not a traceback.
    real_init = cli.Fingerprinter.__init__

    def _patched_init(self: cli.Fingerprinter, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        for handler in self.handlers:

            def _raise(_path: object, _h: object = handler) -> object:
                raise MissingDependencyError(
                    "Pillow is required for image fingerprinting",
                    package="Pillow",
                    extra="image",
                )

            monkeypatch.setattr(handler, "load", _raise)

    monkeypatch.setattr(cli.Fingerprinter, "__init__", _patched_init)

    code, out, err = _run(base + ["fingerprint", str(target)], capsys)

    assert code == 3
    assert out == ""
    assert "error:" in err
    assert "Pillow" in err


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_missing_query_file_exits_1(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _backend_argv(backend, tmp_path)
    missing = tmp_path / "absent.py"

    code, out, err = _run(base + ["search", str(missing)], capsys)

    assert code == 1
    assert out == ""
    assert "error:" in err
    assert "absent.py" in err
