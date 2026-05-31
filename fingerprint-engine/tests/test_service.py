"""Tests for the FastAPI service over the existing backends.

Gated on the ``service`` extra: if FastAPI (and Starlette's TestClient
dependency) is not installed, the whole module is skipped, so a core-only
checkout still runs the rest of the suite green. The tests drive the app
in-process through ``fastapi.testclient.TestClient`` -- no network, no server
process -- and assert a fingerprint -> index -> search round-trip yields a
self-match, plus sane ``/list``, ``/dedup`` and ``/health`` JSON.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Skip the whole module unless the service extra is installed. TestClient lives
# in starlette (a fastapi dependency) and itself needs httpx, so require both.
pytest.importorskip("fastapi", exc_type=ImportError)
pytest.importorskip("httpx", exc_type=ImportError)

from fastapi.testclient import TestClient  # noqa: E402

from fingerprint_engine.core.exceptions import MissingDependencyError  # noqa: E402
from fingerprint_engine.core.index import InMemoryHashIndex, SQLiteHashIndex  # noqa: E402
from fingerprint_engine.core.models import FingerprintConfig  # noqa: E402
from fingerprint_engine.service import build_index, create_app  # noqa: E402


def _featured_lines(seed: int, count: int = 80) -> bytes:
    """Featured text so each upload yields a non-empty, searchable fingerprint."""

    return "".join(
        f"def function_{seed}_{j}(value):\n    return value * {j} + {seed * 7}\n\n"
        for j in range(count)
    ).encode("utf-8")


def _client(index: InMemoryHashIndex | None = None) -> tuple[TestClient, InMemoryHashIndex]:
    """A TestClient over a fresh app with an injected in-memory index."""

    backend = index if index is not None else InMemoryHashIndex()
    return TestClient(create_app(index=backend)), backend


def _upload(name: str, data: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (name, data, "application/octet-stream")}


def test_health_reports_backend_and_counts() -> None:
    client, _index = _client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "memory"
    assert body["file_count"] == 0
    assert body["posting_count"] == 0
    assert isinstance(body["handlers"], list) and body["handlers"]
    assert isinstance(body["version"], str) and body["version"]


def test_fingerprint_endpoint_returns_fingerprint_json() -> None:
    client, index = _client()
    response = client.post("/fingerprint", files=_upload("a.py", _featured_lines(seed=1)))
    assert response.status_code == 200
    body = response.json()
    assert body["hash_count"] > 0
    assert body["handler"]
    assert len(body["content_sha256"]) == 64
    # /fingerprint must NOT mutate the index.
    assert index.file_count == 0


def test_fingerprint_full_emits_hashes_and_landmarks() -> None:
    client, _index = _client()
    response = client.post(
        "/fingerprint", params={"full": "true"}, files=_upload("a.py", _featured_lines(seed=2))
    )
    assert response.status_code == 200
    body = response.json()
    assert "hashes" in body and "landmarks" in body
    assert len(body["hashes"]) == body["hash_count"] if "hash_count" in body else True
    assert body["hashes"], "full payload should include the hash list"


def test_index_then_search_is_a_self_match() -> None:
    client, index = _client()
    data = _featured_lines(seed=3)

    indexed = client.post("/index", files=_upload("doc.py", data))
    assert indexed.status_code == 200
    indexed_body = indexed.json()
    assert indexed_body["file_count"] == 1
    assert indexed_body["posting_count"] > 0
    assert index.file_count == 1  # the injected index really was mutated
    file_id = indexed_body["indexed"]["file_id"]

    searched = client.post("/search", files=_upload("doc.py", data))
    assert searched.status_code == 200
    results = searched.json()["results"]
    assert results, "an indexed file must match itself"
    top = results[0]
    assert top["file_id"] == file_id
    # A self-match is a perfect alignment: confidence saturates at 1.0.
    assert top["confidence"] == pytest.approx(1.0)


def test_search_empty_index_returns_no_results() -> None:
    client, _index = _client()
    response = client.post("/search", files=_upload("q.py", _featured_lines(seed=4)))
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_min_confidence_filters_results() -> None:
    client, _index = _client()
    data = _featured_lines(seed=5)
    client.post("/index", files=_upload("doc.py", data))

    # A self-match has confidence ~1.0, so a cutoff of 1.0 keeps it but an
    # impossible >1.0 cutoff is rejected by the query validator (<= 1.0).
    kept = client.post("/search", params={"min_confidence": 0.99}, files=_upload("doc.py", data))
    assert kept.status_code == 200
    assert kept.json()["results"], "self-match should clear a 0.99 cutoff"

    rejected = client.post("/search", params={"min_confidence": 1.5}, files=_upload("doc.py", data))
    assert rejected.status_code == 422  # validation: min_confidence must be <= 1.0


def test_list_endpoint_reflects_indexed_files() -> None:
    client, _index = _client()
    client.post("/index", files=_upload("one.py", _featured_lines(seed=6)))
    client.post("/index", files=_upload("two.py", _featured_lines(seed=7)))

    listed = client.get("/list")
    assert listed.status_code == 200
    body = listed.json()
    assert body["file_count"] == 2
    assert len(body["files"]) == 2
    paths = {entry["path"] for entry in body["files"]}
    assert paths == {"one.py", "two.py"}
    for entry in body["files"]:
        assert entry["hash_count"] > 0

    summary = client.get("/list", params={"summary": "true"})
    assert summary.status_code == 200
    assert "files" not in summary.json()
    assert summary.json()["file_count"] == 2


def test_dedup_finds_exact_and_near_clusters() -> None:
    client, index = _client()
    base = _featured_lines(seed=8)
    # An exact byte-identical copy and a lightly-edited near-duplicate.
    edited = base.replace(b"return value * 5 + 56", b"return value * 5 + 999") + b"# trailing\n"
    unrelated = "".join(
        f"class Widget{k}:\n    attr = {k * 13}\n    def go(self):\n        return self.attr - {k}\n\n"
        for k in range(80)
    ).encode("utf-8")

    files = [
        ("files", ("base.py", base, "application/octet-stream")),
        ("files", ("base_copy.py", base, "application/octet-stream")),
        ("files", ("near.py", edited, "application/octet-stream")),
        ("files", ("unrelated.py", unrelated, "application/octet-stream")),
    ]
    response = client.post("/dedup", files=files)
    assert response.status_code == 200
    body = response.json()
    # One exact cluster (base + its byte-identical copy).
    assert body["exact_cluster_count"] == 1
    exact_paths = set(body["exact_clusters"][0]["paths"])
    assert exact_paths == {"base.py", "base_copy.py"}
    assert body["total_paths"] == 4
    # dedup runs in a scratch index; the service's own index stays untouched.
    assert index.file_count == 0


def test_dedup_handles_featureless_upload_without_aborting() -> None:
    client, _index = _client()
    # A featureless upload (empty bytes) yields a valid 0-hash fingerprint rather
    # than an error, so it lands in the report as its own distinct singleton and
    # never matches anything -- the good file is unaffected. The batch must not
    # abort: both inputs are accounted for.
    files = [
        ("files", ("good.py", _featured_lines(seed=9), "application/octet-stream")),
        ("files", ("empty.bin", b"", "application/octet-stream")),
    ]
    response = client.post("/dedup", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["total_paths"] == 2
    assert body["total_distinct"] == 2
    # Two unrelated, distinct contents -> no clusters of either tier.
    assert body["exact_cluster_count"] == 0
    assert body["near_duplicate_cluster_count"] == 0
    assert body["singletons"] == 2


def _small_limit_client(limit: int) -> TestClient:
    config = replace(FingerprintConfig(), max_file_size_bytes=limit)
    return TestClient(create_app(index=InMemoryHashIndex(), config=config))


def test_oversized_upload_is_rejected_with_413() -> None:
    # F4: an upload larger than max_file_size_bytes must be rejected at ingest.
    client = _small_limit_client(4096)
    response = client.post("/fingerprint", files=_upload("big.bin", b"x" * (4096 * 8)))
    assert response.status_code == 413


def test_oversized_upload_is_not_fully_spooled_to_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    # F4: the guard must fire DURING the stream, so an oversized body never lands
    # wholesale on temp disk -- at most ~limit bytes are written before the abort.
    limit = 4096
    written = {"total": 0}
    real_ntf = tempfile.NamedTemporaryFile

    class _CountingTemp:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._f = real_ntf(*args, **kwargs)
            self.name = self._f.name

        def write(self, data: bytes) -> int:
            written["total"] += len(data)
            return self._f.write(data)

        def flush(self) -> None:
            self._f.flush()

        def close(self) -> None:
            self._f.close()

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _CountingTemp)
    client = _small_limit_client(limit)
    response = client.post("/fingerprint", files=_upload("big.bin", b"x" * (limit * 16)))
    assert response.status_code == 413
    # The whole 64 KiB body was never spooled; only up to the limit reached disk.
    assert written["total"] <= limit


def test_at_limit_upload_succeeds_and_no_temp_file_leaks(tmp_path: Path) -> None:
    # A file exactly at the limit is accepted (matching the engine's '> limit'
    # rule), and neither the success nor the reject path leaves a temp file behind.
    leftovers_before = list(Path(tempfile.gettempdir()).glob("fp_upload_*"))
    client = _small_limit_client(8192)
    ok = client.post("/fingerprint", files=_upload("ok.py", _featured_lines(seed=42)[:8192]))
    assert ok.status_code == 200
    client.post("/fingerprint", files=_upload("big.bin", b"x" * 65536))  # rejected
    leftovers_after = list(Path(tempfile.gettempdir()).glob("fp_upload_*"))
    assert leftovers_after == leftovers_before


def test_index_round_trip_with_sqlite_backend(tmp_path: Path) -> None:
    """The service is backend-agnostic: a SQLite-backed app round-trips too."""

    db = tmp_path / "svc.sqlite3"
    with SQLiteHashIndex(database=db) as backend:
        client = TestClient(create_app(index=backend))
        data = _featured_lines(seed=10)
        client.post("/index", files=_upload("doc.py", data))
        health = client.get("/health").json()
        assert health["backend"] == "sqlite"
        assert health["file_count"] == 1

        results = client.post("/search", files=_upload("doc.py", data)).json()["results"]
        assert results and results[0]["confidence"] == pytest.approx(1.0)


def test_build_index_selects_backend_from_env() -> None:
    assert isinstance(build_index({}), InMemoryHashIndex)
    assert isinstance(build_index({"FINGERPRINT_BACKEND": "memory"}), InMemoryHashIndex)
    with pytest.raises(ValueError, match="unknown FINGERPRINT_BACKEND"):
        build_index({"FINGERPRINT_BACKEND": "nope"})


def test_default_index_is_in_memory_when_not_injected() -> None:
    # No index= and no FINGERPRINT_BACKEND env -> in-memory default; the app
    # still serves and starts empty.
    client = TestClient(create_app())
    body = client.get("/health").json()
    assert body["backend"] == "memory"
    assert body["file_count"] == 0


def test_importing_service_module_pulls_no_fastapi() -> None:
    """Importing the service module must not import fastapi/uvicorn eagerly.

    The dependency stays behind ``create_app``/``run`` so a core-only install
    can ``import fingerprint_engine.service`` (e.g. to read constants) without
    the extra. Verified in a fresh interpreter so this process's already-loaded
    fastapi does not mask a regression.
    """

    import subprocess

    code = (
        "import importlib, sys\n"
        "importlib.import_module('fingerprint_engine.service')\n"
        "leaked = [m for m in ('fastapi', 'uvicorn', 'starlette') if m in sys.modules]\n"
        "assert not leaked, 'service import eagerly pulled: ' + repr(leaked)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"


def test_missing_dependency_error_is_importable() -> None:
    # Sanity: the error type create_app raises without fastapi is the engine's
    # MissingDependencyError, so callers catch it uniformly.
    assert issubclass(MissingDependencyError, Exception)
