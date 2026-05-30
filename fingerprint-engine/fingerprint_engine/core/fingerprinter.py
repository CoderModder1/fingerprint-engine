"""Fingerprint orchestration and plugin routing."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import os
import pkgutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Literal

from fingerprint_engine.handlers.base import FileHandler

from .exceptions import FileTooLargeError, MissingDependencyError, NoHandlerError
from .fft_pipeline import FFTFingerprintPipeline
from .models import Fingerprint, FingerprintConfig, SearchResult

logger = logging.getLogger(__name__)

# Read files in fixed-size chunks when only the content hash is needed (no
# fingerprinting), so a large file is never pulled fully into memory just to
# learn its sha256. 1 MiB balances syscall count against peak memory.
_SHA_CHUNK_BYTES = 1024 * 1024


def expand_paths(
    paths: Iterable[str | Path],
    *,
    errors: list[tuple[str, Exception]] | None = None,
) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of regular files.

    Each input path is handled independently and fail-soft:

    * a regular file is kept as-is;
    * a directory is walked recursively (``os.walk``) and every regular file
      under it is collected;
    * anything else (a broken symlink, a path that disappears mid-walk, a
      special device, or an unreadable directory) is skipped -- recorded in the
      optional ``errors`` collector as a ``(str(path), exc)`` tuple when one is
      supplied, otherwise simply dropped.

    The result is de-duplicated by resolved path and sorted, so the same
    directory tree always expands to the same ordered file list regardless of
    filesystem iteration order or repeated/overlapping inputs. Symlinks are not
    followed during the directory walk, which prevents both cycle loops and
    escaping the tree being ingested.

    A missing top-level path is surfaced as :class:`FileNotFoundError` in
    ``errors`` (or dropped) rather than raised, so one bad argument never aborts
    the whole expansion -- matching the fail-soft batch philosophy of
    :meth:`Fingerprinter.fingerprint_many`.
    """

    def _record(path_str: str, exc: Exception) -> None:
        logger.debug("expand_paths skipping %s: %s: %s", path_str, type(exc).__name__, exc)
        if errors is not None:
            errors.append((path_str, exc))

    collected: set[Path] = set()
    for raw in paths:
        source = Path(raw)
        try:
            if source.is_dir():
                # followlinks=False (default): never traverse symlinked dirs, so
                # the walk cannot loop on a cycle or escape the tree.
                for dirpath, _dirnames, filenames in os.walk(source):
                    base = Path(dirpath)
                    for name in filenames:
                        candidate = base / name
                        if candidate.is_file():
                            collected.add(candidate)
            elif source.is_file():
                collected.add(source)
            elif not source.exists():
                _record(str(source), FileNotFoundError(source))
            else:
                # Exists but is neither a regular file nor a directory (FIFO,
                # socket, device, broken symlink target): nothing to fingerprint.
                _record(str(source), OSError(f"not a regular file or directory: {source}"))
        except OSError as exc:
            _record(str(source), exc)
    return sorted(collected)


def file_content_sha256(path: str | Path, *, max_file_size_bytes: int = 0) -> str:
    """Compute the sha256 hex digest of a file's bytes, read in chunks.

    This is the same digest the fingerprinter stores as ``content_sha256`` /
    ``file_id`` (``hashlib.sha256`` over the raw file bytes), but computed
    *without* decoding or fingerprinting -- so incremental ingest can learn a
    file's identity cheaply and skip files already present in the index.

    ``max_file_size_bytes`` mirrors :attr:`FingerprintConfig.max_file_size_bytes`:
    when positive, a file larger than the limit raises :class:`FileTooLargeError`
    *before* any bytes are read (the size is taken from ``stat``), bounding the
    work an oversized/hostile input can trigger. 0 means unlimited.
    """

    source = Path(path)
    if max_file_size_bytes > 0:
        size = source.stat().st_size
        if size > max_file_size_bytes:
            raise FileTooLargeError(
                f"{source}: file size {size} bytes exceeds max_file_size_bytes "
                f"limit of {max_file_size_bytes} bytes",
                size=size,
                limit=max_file_size_bytes,
            )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Per-worker Fingerprinter, lazily built by ``_process_worker_init`` so the
# ProcessPoolExecutor reconstructs ONE Fingerprinter per worker process (rather
# than re-pickling the whole instance for every submitted path). On macOS the
# default start method is 'spawn', so workers re-import this module and rebuild
# the Fingerprinter from the picklable (config, handlers_package) pair handed to
# the initializer -- both are trivially importable/picklable.
_WORKER_FINGERPRINTER: Fingerprinter | None = None


def _process_worker_init(config: FingerprintConfig, handlers_package: str) -> None:
    """Initializer run once per worker process: build the shared Fingerprinter.

    Holding the Fingerprinter in a module global keeps fingerprinting itself
    free of any per-task pickling and produces hashes byte-identical to the
    in-process Fingerprinter (same config, same discovered handlers, same
    pipeline) -- the worker is a pure relocation of CPU-bound work, not a
    behavior change.
    """

    global _WORKER_FINGERPRINTER
    _WORKER_FINGERPRINTER = Fingerprinter(config=config, handlers_package=handlers_package)


def _process_worker_fingerprint(path: str | Path) -> Fingerprint:
    """Worker entry point: fingerprint one path with the per-worker instance.

    Top-level (module-scope) so it is importable/picklable under the 'spawn'
    start method. Any exception propagates back through the future exactly as in
    thread mode, preserving the fail-soft / skip_errors semantics in the caller.
    """

    if _WORKER_FINGERPRINTER is None:  # pragma: no cover - initializer always runs first
        raise RuntimeError("process worker was not initialized")
    return _WORKER_FINGERPRINTER.fingerprint_file(path)


class FileProcessor(ABC):
    """Interface for components that produce file fingerprints."""

    @abstractmethod
    def fingerprint_file(self, path: str | Path) -> Fingerprint:
        """Fingerprint a single file."""

    @abstractmethod
    def fingerprint_many(
        self,
        paths: Iterable[str | Path],
        max_workers: int | None = None,
        *,
        skip_errors: bool = True,
        errors: list[tuple[str, Exception]] | None = None,
        executor: Literal["thread", "process"] = "thread",
    ) -> list[Fingerprint]:
        """Fingerprint many files."""


class Fingerprinter(FileProcessor):
    """Routes files through discovered handlers and returns fingerprints."""

    def __init__(
        self,
        config: FingerprintConfig | None = None,
        handlers_package: str = "fingerprint_engine.handlers",
    ) -> None:
        self.config = config or FingerprintConfig()
        self.config.validate()
        self.pipeline = FFTFingerprintPipeline(self.config)
        self.handlers_package = handlers_package
        self.handlers = self.discover_handlers(handlers_package)
        if not self.handlers:
            raise RuntimeError("no file handlers discovered")
        # Handlers are discovered with no constructor args; push config-derived
        # per-handler settings (e.g. the PDF page cap) onto them now.
        for handler in self.handlers:
            handler.configure(self.config)
        self._handler_pipelines = self._build_handler_pipelines()

    def discover_handlers(self, package_name: str) -> list[FileHandler]:
        """Auto-discover FileHandler subclasses in a handler package."""

        package = importlib.import_module(package_name)
        discovered: list[type[FileHandler]] = []
        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            module = importlib.import_module(module_info.name)
            for _name, candidate in inspect.getmembers(module, inspect.isclass):
                if candidate is FileHandler:
                    continue
                if issubclass(candidate, FileHandler):
                    discovered.append(candidate)

        unique: dict[str, type[FileHandler]] = {handler.name: handler for handler in discovered}
        instances = [handler() for handler in unique.values()]
        instances.sort(key=lambda item: (-item.priority, item.name))
        return instances

    def _build_handler_pipelines(self) -> dict[str, FFTFingerprintPipeline]:
        """Per-handler pipelines for handlers that prefer a fixed window/hop.

        A fixed window keeps a content type's fingerprints comparable across
        files of different lengths, so excerpts/truncations of the same content
        still align. The per-handler pipeline is therefore built with
        ``fixed_window=True``: the declared window is *authoritative* and is not
        shrunk as a function of per-file length (which would shift the time grid
        and silently break cross-length matching); it only adapts as a last
        resort for inputs too short to yield a usable frame pair, with a warning.

        Applied only under the default config; an explicitly customized
        window_size/hop_size is honored globally instead (so callers retain full
        control, the global ``--window-size`` override stays length-adaptive,
        and the unit tests' explicit windows are unchanged).
        """

        defaults = FingerprintConfig()
        using_defaults = (
            self.config.window_size == defaults.window_size
            and self.config.hop_size == defaults.hop_size
        )
        pipelines: dict[str, FFTFingerprintPipeline] = {}
        if not using_defaults:
            return pipelines
        for handler in self.handlers:
            window = getattr(handler, "default_signal_window", None)
            if not window:
                continue
            hop = getattr(handler, "default_signal_hop", None) or max(1, window // 4)
            pipelines[handler.name] = FFTFingerprintPipeline(
                replace(self.config, window_size=window, hop_size=hop),
                fixed_window=True,
            )
        return pipelines

    def fingerprint_file(self, path: str | Path) -> Fingerprint:
        """Fingerprint a single file."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        if not source.is_file():
            raise IsADirectoryError(source)

        # Bound the OOM vector from untrusted input: stat the file and reject it
        # BEFORE read_bytes() pulls the whole thing into memory, so a huge file
        # never gets loaded. 0 = unlimited (opt-out). See SECURITY.md.
        limit = self.config.max_file_size_bytes
        if limit > 0:
            size = source.stat().st_size
            if size > limit:
                raise FileTooLargeError(
                    f"{source}: file size {size} bytes exceeds max_file_size_bytes "
                    f"limit of {limit} bytes",
                    size=size,
                    limit=limit,
                )

        content = source.read_bytes()
        content_sha256 = hashlib.sha256(content).hexdigest()
        candidates = self._rank_handlers(source, content[:8192])
        errors: list[str] = []
        top_handler_name = candidates[0][1].name if candidates else None

        for _score, handler in candidates:
            try:
                pipeline = self._handler_pipelines.get(handler.name, self.pipeline)
                payload = handler.load(source)
                signal = handler.to_signal(payload)
                landmarks, hashes = handler.extract_peaks(signal, pipeline)
                effective_window, effective_hop = pipeline.effective_params(signal)
                metadata = handler.metadata(payload)
                metadata.update(
                    {
                        "filename": source.name,
                        "handler_priority": handler.priority,
                        "effective_window_size": effective_window,
                        "effective_hop_size": effective_hop,
                    }
                )
                if not hashes:
                    warnings.warn(
                        f"{source.name}: handler '{handler.name}' produced 0 "
                        "searchable hashes (signal too short or featureless); "
                        "this file will be unsearchable. Try a smaller "
                        "--window-size or a more featured input.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if handler.name != top_handler_name:
                    # A higher-ranked candidate was tried first and failed; the
                    # routing fell back to this one. Surface it so the demotion
                    # is observable without changing behavior.
                    logger.info(
                        "handler %s won for %s after %s higher-ranked candidate(s) failed",
                        handler.name,
                        source,
                        len(errors),
                    )
                return Fingerprint(
                    file_id=content_sha256,
                    path=str(source.resolve()),
                    handler=handler.name,
                    size_bytes=len(content),
                    content_sha256=content_sha256,
                    config=self.config.to_dict(),
                    landmarks=landmarks,
                    hashes=hashes,
                    metadata=metadata,
                )
            except MissingDependencyError as exc:
                # The correct handler exists but its optional dependency is not
                # installed. Re-raise loudly instead of silently demoting to a
                # lower-priority handler (e.g. binary), which would produce
                # fingerprints incomparable to those made with the dependency
                # installed and silently corrupt the index.
                logger.warning(
                    "handler %s for %s is missing optional dependency %s (extra %s)",
                    handler.name,
                    source,
                    exc.package,
                    exc.extra,
                )
                raise
            except Exception as exc:
                logger.debug("handler %s failed for %s: %s", handler.name, source, exc)
                errors.append(f"{handler.name}: {exc}")

        raise NoHandlerError(f"no handler could fingerprint {source}: {'; '.join(errors)}")

    def fingerprint_many(
        self,
        paths: Iterable[str | Path],
        max_workers: int | None = None,
        *,
        skip_errors: bool = True,
        errors: list[tuple[str, Exception]] | None = None,
        executor: Literal["thread", "process"] = "thread",
    ) -> list[Fingerprint]:
        """Fingerprint a batch concurrently while preserving input order.

        Fail-soft by default (``skip_errors=True``): every path is submitted, the
        results are gathered in input order, and any failure (missing file,
        directory, oversized input, no handler, missing dependency, decode error,
        ...) is skipped -- so one bad file never aborts the batch and the good ones
        still come back. With ``skip_errors=False`` the legacy behavior is
        preserved: the first failure propagates.

        Pass ``errors`` to collect failures structurally: each skipped path appends
        a ``(str(path), exc)`` tuple in input order, letting callers (e.g. the CLI)
        report ``type``/``message`` without parsing warning text. When no collector
        is supplied, a :class:`RuntimeWarning` naming the path and error is still
        emitted for callers that rely on it.

        ``executor`` selects the parallelism strategy and is opt-in:

        * ``"thread"`` (default) -- a :class:`ThreadPoolExecutor`, identical to the
          previous behavior. Best for I/O-bound batches and the only mode that
          changes nothing for existing callers.
        * ``"process"`` -- a :class:`ProcessPoolExecutor`. ``build_hashes`` and the
          peak extraction are GIL-bound pure Python, so threads barely parallelize
          CPU-bound fingerprinting; processes give true parallelism. Each worker
          rebuilds one :class:`Fingerprinter` from this instance's
          ``(config, handlers_package)`` via :func:`_process_worker_init`, so the
          produced hashes are byte-identical to thread mode -- only *where* the
          work runs changes, never the output.

        Order preservation, ``skip_errors`` fail-soft semantics, the ``errors``
        collector, and the RuntimeWarning fallback are identical across both modes.
        """

        ordered = list(paths)
        pool, worker = self._build_executor(executor, max_workers)
        with pool:
            futures = [pool.submit(worker, path) for path in ordered]
            if not skip_errors:
                return [future.result() for future in futures]
            results: list[Fingerprint] = []
            for path, future in zip(ordered, futures, strict=True):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - fail-soft: surface and skip
                    logger.warning(
                        "skipping %s: %s: %s", path, type(exc).__name__, exc
                    )
                    if errors is not None:
                        errors.append((str(path), exc))
                    else:
                        warnings.warn(
                            f"skipping {path}: {type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            return results

    def _build_executor(
        self,
        executor: Literal["thread", "process"],
        max_workers: int | None,
    ) -> tuple[Executor, Callable[[str | Path], Fingerprint]]:
        """Resolve the (pool, per-path worker callable) for ``fingerprint_many``.

        The pool is returned unentered so the caller owns its ``with`` block. The
        worker callable is the *only* difference between modes: thread mode calls
        this instance's bound ``fingerprint_file`` directly (shared memory); process
        mode routes through the top-level :func:`_process_worker_fingerprint`, which
        uses the per-worker Fingerprinter built by the initializer. Both produce the
        same Fingerprint for the same path.
        """

        if executor == "thread":
            return ThreadPoolExecutor(max_workers=max_workers), self.fingerprint_file
        if executor == "process":
            pool = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_process_worker_init,
                initargs=(self.config, self.handlers_package),
            )
            return pool, _process_worker_fingerprint
        raise ValueError(f"unknown executor {executor!r}; expected 'thread' or 'process'")

    def search_file(self, path: str | Path, index, top_k: int = 10) -> list[SearchResult]:
        fingerprint = self.fingerprint_file(path)
        return index.search(fingerprint, top_k=top_k)

    def _rank_handlers(
        self,
        path: Path,
        sample: bytes,
    ) -> list[tuple[float, FileHandler]]:
        mime_type = FileHandler.sniff_mime(path)
        scored: list[tuple[float, FileHandler]] = []
        for handler in self.handlers:
            score = handler.can_handle(path, mime_type=mime_type, sample=sample)
            if score > 0:
                scored.append((float(score), handler))
        scored.sort(key=lambda item: (-item[0], -item[1].priority, item[1].name))
        return scored
