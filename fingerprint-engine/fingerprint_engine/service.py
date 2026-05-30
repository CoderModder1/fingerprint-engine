"""Stateless FastAPI service over the existing fingerprint engine.

This module is an ADDITIVE wrapper: it reuses the existing
:class:`~fingerprint_engine.core.fingerprinter.Fingerprinter`,
:class:`~fingerprint_engine.core.index.HashIndex` backends, and
:func:`~fingerprint_engine.core.dedup.find_duplicates` verbatim. No
fingerprinting, indexing, scoring, or ranking logic is reimplemented here, so
exposing the engine over HTTP cannot change hashes or search rankings -- the
service simply marshals uploads in and the existing data structures out.

Dependency boundary
-------------------
``fastapi``/``uvicorn``/``python-multipart`` are *optional* and live behind the
``service`` extra. They are imported lazily inside :func:`create_app` (and the
``__main__`` runner) so that ``import fingerprint_engine`` and even ``import
fingerprint_engine.service`` stay dependency-free for a core-only install. When
FastAPI is absent, :func:`create_app` raises a clear
:class:`~fingerprint_engine.core.exceptions.MissingDependencyError` naming the
extra to install.

Concurrency contract
---------------------
One :class:`Fingerprinter` is constructed at app startup and reused across all
requests (fingerprinting is pure/stateless per call, so it is safe to share).
Within a single process this is concurrency-safe even though Starlette dispatches
the sync endpoint handlers to a threadpool: the in-memory backend's reads are
lock-free and its writes lock-serialized, and the SQLite/PostgreSQL backends
serialize ALL connection access under a per-index re-entrant lock (so concurrent
``/search`` requests can no longer interleave the shared SQLite scratch table).
The index follows the engine's Tier-2 concurrency contract: a SINGLE logical
writer. ``POST /index`` mutates the index; the in-memory backend serializes
mutations under its own re-entrant lock, while the SQLite/PostgreSQL backends
are *per-process* connections (a backend object opened in one worker process is
not shared with another). Run the service as a SINGLE process (the default
``uvicorn`` invocation, ``--workers 1``) when writing through ``/index``; for
read-heavy multi-worker deployments, point each worker at a shared persistent
backend (Redis/PostgreSQL) and treat writes as funneled through one writer.
The injected ``index`` is owned by the app for its lifetime.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .core.dedup import DEFAULT_MIN_CONFIDENCE, find_duplicates
from .core.exceptions import (
    FileTooLargeError,
    MissingDependencyError,
    NoHandlerError,
)
from .core.fingerprinter import Fingerprinter
from .core.index import (
    HashIndex,
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from .core.models import Calibration, Fingerprint, FingerprintConfig

if TYPE_CHECKING:  # pragma: no cover - typing only; never imported at runtime
    # The real types, for static checking only. At runtime these names are
    # injected into module globals by _load_fastapi (see below) so the string
    # endpoint annotations resolve; under TYPE_CHECKING they come from fastapi.
    from fastapi import FastAPI, UploadFile
    from fastapi.responses import JSONResponse


def _load_fastapi() -> dict[str, Any]:
    """Lazily import the FastAPI symbols and inject ``UploadFile`` into globals.

    Returns the imported ``{FastAPI, File, Query, JSONResponse, UploadFile}`` for
    use as locals in :func:`create_app`, and -- crucially -- also binds
    ``UploadFile`` as a MODULE global. With ``from __future__ import annotations``
    in force, every endpoint annotation is a *string* that FastAPI resolves via
    ``typing.get_type_hints`` against the endpoint function's module globals, so
    ``UploadFile`` must be a real name in THIS module's namespace at request
    time, not merely a local closed over by ``create_app``. The import stays
    lazy (this runs only when the service is actually built), preserving the
    dependency-free top-level import contract.

    Raises :class:`MissingDependencyError` naming the ``service`` extra when
    FastAPI is not installed.
    """

    try:
        from fastapi import FastAPI, File, Query, UploadFile
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised only without fastapi
        raise MissingDependencyError(
            "the HTTP service requires FastAPI (and uvicorn to serve); install "
            "them with 'pip install \"fingerprint-engine[service]\"'",
            package="fastapi",
            extra="service",
        ) from exc
    # Bind into module globals so get_type_hints can resolve the "UploadFile"
    # string annotations on the endpoint functions defined inside create_app.
    globals()["UploadFile"] = UploadFile
    return {
        "FastAPI": FastAPI,
        "File": File,
        "Query": Query,
        "JSONResponse": JSONResponse,
        "UploadFile": UploadFile,
    }


# Environment variable names honored by :func:`build_index` when no index is
# injected. Mirrors the CLI's backend selection so the service and CLI agree.
ENV_BACKEND = "FINGERPRINT_BACKEND"
ENV_SQLITE_PATH = "FINGERPRINT_SQLITE_PATH"
ENV_REDIS_URL = "FINGERPRINT_REDIS_URL"
ENV_REDIS_PREFIX = "FINGERPRINT_REDIS_PREFIX"
ENV_POSTGRES_DSN = "FINGERPRINT_POSTGRES_DSN"

_DEFAULT_SQLITE_PATH = ".fingerprint_index.sqlite3"
_DEFAULT_REDIS_URL = "redis://localhost:6379/0"
_DEFAULT_REDIS_PREFIX = "fpidx"
_DEFAULT_POSTGRES_DSN = "postgresql://localhost/fingerprint"


def build_index(env: dict[str, str] | None = None) -> HashIndex:
    """Construct the index backend selected by the environment.

    ``FINGERPRINT_BACKEND`` chooses the backend (``memory`` [default],
    ``sqlite``, ``redis``, ``postgres``); the backend-specific connection
    settings come from the matching ``FINGERPRINT_*`` variables. This mirrors
    the CLI's :func:`fingerprint_engine.cli.open_index` so a service and a CLI
    pointed at the same configuration share one store. An explicit ``index=``
    passed to :func:`create_app` takes precedence and bypasses this entirely.
    """

    source = os.environ if env is None else env
    backend = source.get(ENV_BACKEND, "memory").lower()
    if backend == "memory":
        return InMemoryHashIndex()
    if backend == "sqlite":
        return SQLiteHashIndex(database=source.get(ENV_SQLITE_PATH, _DEFAULT_SQLITE_PATH))
    if backend == "redis":
        return RedisHashIndex(
            url=source.get(ENV_REDIS_URL, _DEFAULT_REDIS_URL),
            key_prefix=source.get(ENV_REDIS_PREFIX, _DEFAULT_REDIS_PREFIX),
        )
    if backend == "postgres":
        return PostgresHashIndex(dsn=source.get(ENV_POSTGRES_DSN, _DEFAULT_POSTGRES_DSN))
    raise ValueError(
        f"unknown {ENV_BACKEND}={backend!r}; expected one of "
        "'memory', 'sqlite', 'redis', 'postgres'"
    )


def _summarize_fingerprint(fingerprint: Fingerprint, *, full: bool = False) -> dict[str, Any]:
    """JSON view of a fingerprint, matching the CLI's ``summarize_fingerprint``.

    ``full=True`` emits the complete hash/landmark payload (via
    :meth:`Fingerprint.to_dict`); otherwise a compact summary with a sample of
    the leading hashes -- the same trimmed shape the CLI prints, kept in sync so
    HTTP and CLI consumers see identical fields.
    """

    if full:
        return fingerprint.to_dict(include_landmarks=True)
    return {
        "file_id": fingerprint.file_id,
        "path": fingerprint.path,
        "handler": fingerprint.handler,
        "size_bytes": fingerprint.size_bytes,
        "content_sha256": fingerprint.content_sha256,
        "landmark_count": fingerprint.landmark_count,
        "hash_count": fingerprint.hash_count,
        "sample_hashes": [
            {"hash_code": item.hash_code, "time_offset": item.time_offset}
            for item in fingerprint.hashes[:10]
        ],
        "metadata": fingerprint.metadata,
    }


def create_app(
    index: HashIndex | None = None,
    config: FingerprintConfig | None = None,
) -> FastAPI:
    """Build a stateless FastAPI app exposing the fingerprint engine over HTTP.

    Parameters
    ----------
    index:
        The :class:`HashIndex` backend to read/write. When ``None`` (the
        default) a backend is constructed from the ``FINGERPRINT_*`` environment
        variables via :func:`build_index` (an :class:`InMemoryHashIndex` unless
        overridden). The app owns the injected index for its lifetime.
    config:
        The :class:`FingerprintConfig` for the single shared
        :class:`Fingerprinter`. ``None`` uses the engine defaults -- so default
        fingerprints/rankings are byte-identical to the CLI and library.

    Routes (all uploads are ``multipart/form-data`` with a ``file`` field):

    * ``POST /fingerprint`` -- fingerprint an upload, return its fingerprint JSON
      (``?full=true`` for the complete hash/landmark payload).
    * ``POST /index`` -- fingerprint an upload and add it to the index.
    * ``POST /search`` -- fingerprint a query upload and return ranked results
      (``?top_k=`` and optional ``?min_confidence=`` in [0, 1]).
    * ``GET /list`` -- list indexed files (``?summary=true`` for counts only).
    * ``POST /dedup`` -- cluster several uploaded files into exact/near groups.
    * ``GET /health`` -- liveness/readiness probe with backend + counts.

    FastAPI is imported lazily here; a core-only install raises
    :class:`MissingDependencyError` naming the ``service`` extra to install.
    """

    fastapi_symbols = _load_fastapi()
    # Local handles to the lazily-imported callables. ``File``/``Query`` are used
    # as request-parameter dependency markers (default values, evaluated at def
    # time); ``json_response`` builds error responses. The annotation NAMES
    # (UploadFile/JSONResponse) resolve via module globals / TYPE_CHECKING.
    fastapi_cls = fastapi_symbols["FastAPI"]
    File = fastapi_symbols["File"]
    Query = fastapi_symbols["Query"]
    json_response = fastapi_symbols["JSONResponse"]

    # ONE Fingerprinter, shared across every request. Default config keeps hashes
    # byte-identical to the library/CLI; a caller-supplied config customizes the
    # whole service. The index is resolved once at construction.
    fingerprinter = Fingerprinter(config or FingerprintConfig())
    active_index: HashIndex = index if index is not None else build_index()

    app = fastapi_cls(
        title="fingerprint-engine",
        description="Universal file fingerprinting (Shazam-style) over HTTP.",
        version=_package_version(),
    )

    def _fingerprint_upload(upload: UploadFile) -> Fingerprint:
        """Spool an upload to a temp file and fingerprint it via the engine.

        The :class:`Fingerprinter` reads from a path (it stats the file to
        enforce ``max_file_size_bytes`` before reading), so the streamed upload
        is written to a NamedTemporaryFile, fingerprinted, then deleted. The
        original client filename is restored on the result so ``metadata`` /
        ``path`` reflect the upload rather than the throwaway temp name.

        The ``max_file_size_bytes`` limit is enforced DURING the stream (F4): the
        engine's stat-based guard only fires AFTER the whole body is on disk, so
        without an in-loop check an oversized upload would fill temp disk and burn
        worker time before being rejected. Here a running byte count aborts as soon
        as the limit is exceeded, so at most ``limit`` bytes ever reach disk; the
        same :class:`FileTooLargeError` is raised, mapped to 413 exactly as before.
        For any in-limit upload the spooled bytes are identical, so the resulting
        fingerprint is unchanged.
        """

        limit = fingerprinter.config.max_file_size_bytes  # 0 = unlimited
        suffix = Path(upload.filename or "").suffix
        tmp = tempfile.NamedTemporaryFile(prefix="fp_upload_", suffix=suffix, delete=False)
        try:
            # Stream the body to disk in chunks so a large upload is never held
            # fully in memory here, counting bytes so the size guard fires at
            # ingest rather than after the whole (possibly hostile) body is spooled.
            written = 0
            while True:
                # Read no more than one byte past the limit so an over-limit upload
                # is detected without pulling an extra full chunk into memory.
                to_read = 1024 * 1024 if limit <= 0 else min(1024 * 1024, limit - written + 1)
                chunk = upload.file.read(to_read)
                if not chunk:
                    break
                written += len(chunk)
                if limit > 0 and written > limit:
                    tmp.close()
                    raise FileTooLargeError(
                        f"upload exceeds max_file_size_bytes limit of {limit} bytes",
                        size=written,
                        limit=limit,
                    )
                tmp.write(chunk)
            tmp.flush()
            tmp.close()
            fingerprint = fingerprinter.fingerprint_file(tmp.name)
        finally:
            os.unlink(tmp.name)
        # Reflect the client-supplied name, not the temp path, in the result.
        client_name = upload.filename or fingerprint.file_id
        fingerprint.path = client_name
        fingerprint.metadata = {**fingerprint.metadata, "filename": Path(client_name).name}
        return fingerprint

    def _error(status_code: int, detail: str) -> JSONResponse:
        return json_response(status_code=status_code, content={"detail": detail})

    def _handle_fingerprint_errors(upload: UploadFile) -> Fingerprint | JSONResponse:
        """Fingerprint an upload, mapping engine errors to JSON HTTP responses.

        Mirrors the CLI's exit-code mapping: a missing optional dependency is a
        502-style backend gap (the right handler exists but its extra is not
        installed), a too-large or unhandleable input is a 4xx client error.
        """

        try:
            return _fingerprint_upload(upload)
        except MissingDependencyError as exc:
            return _error(503, str(exc))
        except FileTooLargeError as exc:
            return _error(413, str(exc))
        except NoHandlerError as exc:
            return _error(415, str(exc))

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness/readiness: confirms the shared Fingerprinter and index work."""

        return {
            "status": "ok",
            "version": _package_version(),
            "handlers": [handler.name for handler in fingerprinter.handlers],
            "backend": _backend_name(active_index),
            "file_count": active_index.file_count,
            "posting_count": active_index.posting_count,
        }

    @app.post("/fingerprint")
    def fingerprint_endpoint(
        file: UploadFile = File(...),
        full: bool = Query(False, description="emit full hashes and landmarks"),
    ) -> Any:
        """Fingerprint an upload and return its fingerprint JSON (not indexed)."""

        result = _handle_fingerprint_errors(file)
        if not isinstance(result, Fingerprint):
            return result  # an error JSONResponse
        return _summarize_fingerprint(result, full=full)

    @app.post("/index")
    def index_endpoint(file: UploadFile = File(...)) -> Any:
        """Fingerprint an upload and add it to the index (the single writer)."""

        result = _handle_fingerprint_errors(file)
        if not isinstance(result, Fingerprint):
            return result  # an error JSONResponse
        active_index.add(result)
        return {
            "indexed": _summarize_fingerprint(result, full=False),
            "backend": _backend_name(active_index),
            "file_count": active_index.file_count,
            "posting_count": active_index.posting_count,
        }

    @app.post("/search")
    def search_endpoint(
        file: UploadFile = File(...),
        top_k: int = Query(10, ge=1, description="maximum number of ranked results"),
        min_confidence: float | None = Query(
            None,
            ge=0.0,
            le=1.0,
            description="drop matches below this confidence in [0, 1] (handler-comparable)",
        ),
    ) -> Any:
        """Fingerprint a query upload and return ranked matches from the index."""

        result = _handle_fingerprint_errors(file)
        if not isinstance(result, Fingerprint):
            return result  # an error JSONResponse
        calibration = (
            Calibration(default_min_confidence=min_confidence)
            if min_confidence is not None
            else None
        )
        results = active_index.search(result, top_k=top_k, calibration=calibration)
        return {
            "query": _summarize_fingerprint(result, full=False),
            "backend": _backend_name(active_index),
            "results": [match.to_dict() for match in results],
        }

    @app.get("/list")
    def list_endpoint(
        summary: bool = Query(False, description="emit only the counts, omit the file list"),
    ) -> dict[str, Any]:
        """List indexed files (streamed per-file metadata), or just the counts."""

        payload: dict[str, Any] = {
            "backend": _backend_name(active_index),
            "file_count": active_index.file_count,
            "posting_count": active_index.posting_count,
        }
        if not summary:
            payload["files"] = [
                {
                    "file_id": str(meta.get("file_id", "")),
                    "path": str(meta.get("path", "")),
                    "handler": str(meta.get("handler", "")),
                    "hash_count": int(meta.get("hash_count", 0) or 0),
                }
                for meta in active_index.iter_metadata()
            ]
        return payload

    @app.post("/dedup")
    def dedup_endpoint(
        files: list[UploadFile] = File(...),
        min_confidence: float = Query(
            DEFAULT_MIN_CONFIDENCE,
            ge=0.0,
            le=1.0,
            description="near-duplicate confidence cutoff in [0, 1] (higher = stricter)",
        ),
    ) -> Any:
        """Cluster several uploads into exact and near-duplicate groups.

        Each upload is fingerprinted once (fail-soft: an unhandleable/too-large
        upload is reported under ``skipped`` rather than aborting the batch),
        then :func:`find_duplicates` clusters them in a SCRATCH in-memory index
        -- it never touches the service's persistent backend, matching the CLI's
        dedup semantics.
        """

        fingerprints: list[Fingerprint] = []
        skipped: list[dict[str, str]] = []
        for upload in files:
            try:
                fingerprints.append(_fingerprint_upload(upload))
            except (MissingDependencyError, FileTooLargeError, NoHandlerError) as exc:
                skipped.append(
                    {
                        "filename": upload.filename or "",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        report = find_duplicates(fingerprints, min_confidence=min_confidence)
        return {"min_confidence": min_confidence, "skipped": skipped, **report.to_dict()}

    return app


def _backend_name(index: HashIndex) -> str:
    """A short, stable backend label for responses (no postings materialized).

    Uses the class name rather than :meth:`HashIndex.to_dict` (which would
    serialize the whole index) so ``/health`` and ``/list`` stay O(1).
    """

    mapping = {
        "InMemoryHashIndex": "memory",
        "SQLiteHashIndex": "sqlite",
        "RedisHashIndex": "redis",
        "PostgresHashIndex": "postgres",
    }
    return mapping.get(type(index).__name__, type(index).__name__)


def _package_version() -> str:
    """The installed package version, or the source-tree fallback."""

    from . import __version__

    return __version__


def _config_from_env(env: dict[str, str] | None = None) -> FingerprintConfig | None:
    """Build a :class:`FingerprintConfig` override from ``FINGERPRINT_*`` env vars.

    Returns ``None`` (engine defaults) unless a known tuning variable is set, so
    the default service path stays byte-identical to the library. Only a small,
    safe subset is exposed via env; full control is available by calling
    :func:`create_app` with an explicit ``config``.
    """

    source = os.environ if env is None else env
    # dict[str, Any] (not dict[str, int]) so ``replace(..., **overrides)`` type-checks:
    # FingerprintConfig now has non-int fields (e.g. the opt-in window_bank tuple),
    # and mypy validates a splatted dict against the union of *all* field types. The
    # values inserted here are still ints; this annotation has no runtime effect.
    overrides: dict[str, Any] = {}
    if (window := source.get("FINGERPRINT_WINDOW_SIZE")) is not None:
        overrides["window_size"] = int(window)
    if (hop := source.get("FINGERPRINT_HOP_SIZE")) is not None:
        overrides["hop_size"] = int(hop)
    if (max_size := source.get("FINGERPRINT_MAX_FILE_SIZE")) is not None:
        overrides["max_file_size_bytes"] = int(max_size)
    if not overrides:
        return None
    return replace(FingerprintConfig(), **overrides)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover - thin runner
    """Serve the app with uvicorn (single process; honors ``FINGERPRINT_*`` env).

    A minimal entry point so ``python -m fingerprint_engine.service`` brings up
    the service. uvicorn is imported lazily here (same ``service`` extra). Run a
    SINGLE worker when writing through ``/index`` (see the module concurrency
    note); the SQLite/PostgreSQL backends are per-process.
    """

    try:
        import uvicorn
    except ImportError as exc:
        raise MissingDependencyError(
            "serving requires uvicorn; install it with "
            "'pip install \"fingerprint-engine[service]\"'",
            package="uvicorn",
            extra="service",
        ) from exc
    app = create_app(config=_config_from_env())
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover - module runner
    run(
        host=os.environ.get("FINGERPRINT_HOST", "127.0.0.1"),
        port=int(os.environ.get("FINGERPRINT_PORT", "8000")),
    )
