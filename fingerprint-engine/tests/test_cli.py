from __future__ import annotations

import json
import platform
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

    # A8: the counts block reconciles even when an argument fails to expand --
    # scanned counts the true input population (the failed arg included).
    counts = payload["counts"]
    assert counts["scanned"] == counts["skipped_existing"] + counts["newly_indexed"] + counts["failed"]
    assert counts["scanned"] == 4  # 3 good files + 1 missing
    assert counts["failed"] == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_counts_reconcile_with_incremental(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A8: the identity scanned == skipped_existing + newly_indexed + failed holds
    # in --incremental mode too, with a good file + a nonexistent argument.
    good = _write_corpus(tmp_path)
    base = _backend_argv(backend, tmp_path)
    # Seed the index so a re-add is counted as skipped_existing.
    assert _run(base + ["add", str(good[0])], capsys)[0] == 0
    missing = tmp_path / "ghost.py"
    code, out, _err = _run(
        base + ["add", "--incremental", str(good[0]), str(good[1]), str(missing)], capsys
    )
    assert code == 0
    counts = json.loads(out)["counts"]
    assert counts["scanned"] == counts["skipped_existing"] + counts["newly_indexed"] + counts["failed"]
    assert counts["scanned"] == 3  # good[0] (existing) + good[1] (new) + missing (failed)
    assert counts["skipped_existing"] == 1
    assert counts["newly_indexed"] == 1
    assert counts["failed"] == 1


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

            def _raise(_path: object, *, content: bytes | None = None, _h: object = handler) -> object:
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_list_command_emits_indexed_files(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _write_corpus(tmp_path)
    base = _backend_argv(backend, tmp_path)

    # An empty index lists zero files.
    code, out, _err = _run(base + ["list"], capsys)
    assert code == 0
    empty = json.loads(out)
    assert empty["backend"] == backend
    assert empty["file_count"] == 0
    assert empty["files"] == []

    add_code, add_out, _ = _run(base + ["add", *[str(p) for p in good]], capsys)
    assert add_code == 0
    added = {
        item["file_id"]: Path(item["path"]).name for item in json.loads(add_out)["indexed_files"]
    }

    code, out, _err = _run(base + ["list"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["backend"] == backend
    assert payload["file_count"] == 3
    # Each listed file carries the stable projection fields, sorted by file_id.
    listed_ids = [item["file_id"] for item in payload["files"]]
    assert listed_ids == sorted(added)
    assert {item["path"] and Path(item["path"]).name for item in payload["files"]} == set(added.values())
    for item in payload["files"]:
        assert set(item) == {"file_id", "path", "handler", "hash_count"}
        assert item["handler"] == "text"
        assert item["hash_count"] > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_list_summary_omits_per_file_list(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _write_corpus(tmp_path)
    base = _backend_argv(backend, tmp_path)
    assert _run(base + ["add", *[str(p) for p in good]], capsys)[0] == 0

    code, out, _err = _run(base + ["list", "--summary"], capsys)

    assert code == 0
    payload = json.loads(out)
    assert payload["file_count"] == 3
    assert payload["posting_count"] > 0
    assert "files" not in payload  # --summary drops the per-file list


def _write_corpus_in(directory: Path, count: int = 3) -> list[Path]:
    """A featured-text corpus written under ``directory`` (created if needed).

    The directory name is folded into each file's body so that same-named files
    in different directories have DISTINCT content (distinct content_sha256, the
    index's file_id) and are therefore indexed as distinct files rather than
    collapsed by content-dedup.
    """

    directory.mkdir(parents=True, exist_ok=True)
    tag = directory.name
    paths: list[Path] = []
    for i in range(count):
        path = directory / f"doc_{i}.py"
        lines = [
            f"def function_{tag}_{i}_{j}(value):\n    return value * {j} + {i * 7}\n\n"
            for j in range(60)
        ]
        path.write_text("".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_directory_is_walked_recursively(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Files spread across a nested directory tree; `add <dir>` must collect them
    # all, including those in the sub-directory, from a single directory argument.
    corpus = tmp_path / "corpus"
    top = _write_corpus_in(corpus, count=2)
    nested = _write_corpus_in(corpus / "sub", count=2)
    base = _backend_argv(backend, tmp_path)

    code, out, _err = _run(base + ["add", str(corpus)], capsys)

    assert code == 0
    payload = json.loads(out)
    indexed_names = {Path(item["path"]).name for item in payload["indexed_files"]}
    # doc_0/doc_1 appear in both the top dir and the sub dir; same names, distinct
    # content (different bytes), so all four are indexed as distinct files.
    assert len(payload["indexed_files"]) == len(top) + len(nested) == 4
    assert indexed_names == {"doc_0.py", "doc_1.py"}
    assert payload["counts"]["scanned"] == 4
    assert payload["counts"]["newly_indexed"] == 4
    assert payload["counts"]["failed"] == 0
    assert payload["file_count"] == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_incremental_skips_on_second_run_and_adds_only_new(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    initial = _write_corpus_in(corpus, count=3)
    base = _backend_argv(backend, tmp_path)

    # First incremental run over the directory indexes every file.
    code1, out1, _ = _run(base + ["add", str(corpus), "--incremental"], capsys)
    assert code1 == 0
    first = json.loads(out1)
    assert first["counts"]["scanned"] == 3
    assert first["counts"]["skipped_existing"] == 0
    assert first["counts"]["newly_indexed"] == 3
    assert first["counts"]["failed"] == 0
    assert first["file_count"] == 3

    # Second incremental run over the SAME directory skips all of them: nothing
    # new is fingerprinted, the index is unchanged.
    code2, out2, _ = _run(base + ["add", str(corpus), "--incremental"], capsys)
    assert code2 == 0
    second = json.loads(out2)
    assert second["counts"]["scanned"] == 3
    assert second["counts"]["skipped_existing"] == 3
    assert second["counts"]["newly_indexed"] == 0
    assert second["indexed_files"] == []
    assert second["file_count"] == 3  # unchanged

    # Add ONE new file, re-run incrementally: only the new file is indexed.
    new_file = corpus / "doc_new.py"
    new_file.write_text(
        "".join(
            f"def brand_new_{j}(value):\n    return value - {j} * 13\n\n" for j in range(60)
        ),
        encoding="utf-8",
    )
    code3, out3, _ = _run(base + ["add", str(corpus), "--incremental"], capsys)
    assert code3 == 0
    third = json.loads(out3)
    assert third["counts"]["scanned"] == 4
    assert third["counts"]["skipped_existing"] == 3
    assert third["counts"]["newly_indexed"] == 1
    indexed_names = {Path(item["path"]).name for item in third["indexed_files"]}
    assert indexed_names == {"doc_new.py"}
    assert third["file_count"] == 4

    # The --skip-existing alias selects the same behavior.
    code4, out4, _ = _run(base + ["add", str(corpus), "--skip-existing"], capsys)
    assert code4 == 0
    fourth = json.loads(out4)
    assert fourth["counts"]["skipped_existing"] == 4
    assert fourth["counts"]["newly_indexed"] == 0

    _ = initial  # corpus authored above; referenced for clarity


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_incremental_directory_with_oversized_file_succeeds_fail_soft(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A directory holding good files plus one oversized input: the add must
    # still succeed (exit 0), index the good ones, and report the oversized one
    # as a failure (FileTooLargeError, rejected before its bytes are read on the
    # cheap sha path) without aborting the batch.
    corpus = tmp_path / "corpus"
    good = _write_corpus_in(corpus, count=2)
    big = corpus / "huge.bin"
    big.write_bytes(b"\x00" * 200_000)  # over the 100000-byte cap below
    base = _backend_argv(backend, tmp_path)

    code, out, _err = _run(
        base + ["--max-file-size", "100000", "add", str(corpus), "--incremental"], capsys
    )

    assert code == 0
    payload = json.loads(out)
    indexed_names = {Path(item["path"]).name for item in payload["indexed_files"]}
    assert indexed_names == {p.name for p in good}
    assert payload["file_count"] == len(good) == 2
    # scanned counts every regular file found (good + oversized); newly_indexed
    # counts only the successfully fingerprinted good files; the oversized file
    # is reported as a structured failure.
    assert payload["counts"]["scanned"] == 3
    assert payload["counts"]["newly_indexed"] == 2
    assert payload["counts"]["failed"] == 1
    bad_reasons = [item["reason"] for item in payload["skipped"]]
    assert any(reason.startswith("FileTooLargeError") for reason in bad_reasons)


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_non_incremental_default_reindexes_existing(
    backend: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without --incremental, the default behavior is unchanged: every scanned
    # file is (re-)fingerprinted and added, even when already present.
    corpus = tmp_path / "corpus"
    _write_corpus_in(corpus, count=3)
    base = _backend_argv(backend, tmp_path)

    code1, out1, _ = _run(base + ["add", str(corpus)], capsys)
    assert code1 == 0
    assert json.loads(out1)["counts"]["newly_indexed"] == 3

    code2, out2, _ = _run(base + ["add", str(corpus)], capsys)
    assert code2 == 0
    second = json.loads(out2)
    # Default add re-fingerprints all three (no skip); file_count stays 3 because
    # the index dedupes by content sha (last write wins).
    assert second["counts"]["newly_indexed"] == 3
    assert second["counts"]["skipped_existing"] == 0
    assert second["file_count"] == 3


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="no-subcommand"),
        pytest.param(["not-a-command"], id="unknown-subcommand"),
        pytest.param(["search"], id="missing-required-positional"),
        pytest.param(["search", "query.py", "--top-k", "not-an-int"], id="bad-int-type"),
        pytest.param(["--backend", "invalid", "list"], id="bad-backend-choice"),
    ],
)
def test_usage_errors_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    # argparse usage errors are raised as SystemExit(2) from parse_args, BEFORE
    # main()'s try/except runs -- so they surface as the conventional exit code 2
    # (a usage error), distinct from the library exit codes 1/3/4. main() never
    # catches SystemExit, so the code propagates unchanged.
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == 2
    # argparse writes its diagnostic to stderr, never stdout.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


def test_prune_operational_error_exits_4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A backend that declines an operation (here Redis, whose prune_stop_hashes
    # raises NotImplementedError -- a RuntimeError subclass) must surface as the
    # operational exit code 4, not a traceback. open_index is stubbed so the test
    # needs no live Redis: with the real driver absent, open_index would instead
    # raise MissingDependencyError (exit 3) before prune is ever reached.
    class _DecliningIndex:
        def prune_stop_hashes(self, max_df_ratio: float = 0.1) -> int:
            raise NotImplementedError(
                "RedisHashIndex does not support prune_stop_hashes; "
                "rebuild from a snapshot of a pruned index instead"
            )

    monkeypatch.setattr(cli, "open_index", lambda args: _DecliningIndex())

    code, out, err = _run(
        ["--backend", "redis", "prune", "--max-df-ratio", "0.1"], capsys
    )

    assert code == 4
    assert out == ""
    assert "error:" in err
    assert "prune_stop_hashes" in err


def test_corrupt_index_snapshot_exits_4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unreadable/corrupt memory snapshot is an operational failure (a
    # FingerprintError from the index layer), not a usage error: exit 4, clean
    # one-line stderr, no traceback.
    bad_index = tmp_path / "corrupt.json"
    bad_index.write_text("{ this is not valid json", encoding="utf-8")
    query = _write_corpus(tmp_path)[0]

    code, out, err = _run(
        ["--index-path", str(bad_index), "search", str(query)], capsys
    )

    assert code == 4
    assert out == ""
    assert "error:" in err


def test_doctor_exits_0_and_reports_versions_and_capabilities(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The deps doctor is a pure environment diagnosis: it never touches a
    # backend, so it runs without --index-path/--backend and always exits 0.
    code, out, err = _run(["doctor"], capsys)

    assert code == 0
    assert err == ""
    payload = json.loads(out)

    # Versions: the running interpreter and the installed engine version.
    from fingerprint_engine import __version__

    assert payload["python_version"] == platform.python_version()
    assert payload["fingerprint_engine_version"] == __version__

    # Core capabilities are always available (numpy-only install).
    assert payload["core"]["requires"] == ["numpy"]
    assert set(payload["core"]["handlers"]) == {"binary", "text", "archive", "embedding"}
    assert set(payload["core"]["backends"]) == {"memory", "sqlite"}
    # The core handlers/backends are always reported as available.
    assert {"binary", "text", "archive"}.issubset(set(payload["available_handlers"]))
    assert {"memory", "sqlite"}.issubset(set(payload["available_backends"]))


def test_doctor_reports_every_optional_extra(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Every extra declared in pyproject's optional-dependencies (minus the
    # aggregate `all`/`dev`) must appear with a stable shape: an availability
    # flag, the modules it requires, per-module import probes, and the
    # handlers/backends/services it unlocks.
    code, out, _err = _run(["doctor"], capsys)
    assert code == 0
    extras = json.loads(out)["extras"]

    assert set(extras) == {
        "image",
        "audio",
        "pdf",
        "video",
        "embeddings",
        "redis",
        "postgres",
        "service",
    }
    for name, report in extras.items():
        assert set(report) == {
            "available",
            "requires",
            "optional",
            "probes",
            "handlers",
            "backends",
            "services",
            "encoders",
        }
        assert isinstance(report["available"], bool)
        assert report["requires"], f"{name} must list its required modules"
        # One probe per (required + optional) module, in that order, each with a
        # stable shape. `requires` gates availability; `optional` only adds a
        # capability (e.g. audio's pydub/MP3), so it is probed but not required.
        assert [probe["module"] for probe in report["probes"]] == (
            report["requires"] + report["optional"]
        )
        for probe in report["probes"]:
            assert isinstance(probe["ok"], bool)
            if not probe["ok"]:
                assert probe["error"]  # a failed probe explains why
        # available iff every REQUIRED module imported (optional modules do not gate).
        required_ok = {
            probe["module"]: probe["ok"]
            for probe in report["probes"]
            if probe["module"] in report["requires"]
        }
        assert report["available"] == all(required_ok[m] for m in report["requires"])

    # The capability mapping is correct: these extras unlock these names.
    assert extras["image"]["handlers"] == ["image"]
    assert extras["audio"]["handlers"] == ["audio"]
    assert extras["audio"]["optional"] == ["pydub"]
    assert extras["pdf"]["handlers"] == ["pdf"]
    assert extras["video"]["handlers"] == ["video"]
    assert extras["embeddings"]["encoders"] == ["model2vec"]
    assert extras["redis"]["backends"] == ["redis"]
    assert extras["postgres"]["backends"] == ["postgres"]
    assert extras["service"]["services"] == ["http"]


def test_doctor_capability_lists_follow_availability(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force every optional import to fail: the doctor must still exit 0 and
    # report ONLY the core handlers/backends as available, with each extra
    # marked unavailable and carrying the import error.
    real_import = cli.importlib.import_module
    optional = {
        "PIL", "scipy", "pydub", "pypdf", "av", "model2vec", "redis", "psycopg",
        "fastapi", "uvicorn", "python_multipart",
    }

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in optional:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", _fake_import)

    code, out, err = _run(["doctor"], capsys)

    assert code == 0
    assert err == ""
    payload = json.loads(out)
    # With no extras importable, only the core capabilities remain available.
    # The embedding handler's precomputed path is numpy-only, so it is core.
    assert payload["available_handlers"] == sorted({"binary", "text", "archive", "embedding"})
    assert payload["available_backends"] == sorted({"memory", "sqlite"})
    assert payload["available_encoders"] == []
    for name, report in payload["extras"].items():
        assert report["available"] is False, f"{name} must be unavailable when its deps fail to import"
        assert all(probe["ok"] is False and probe["error"] for probe in report["probes"])
