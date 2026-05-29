"""Fingerprint orchestration and plugin routing."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import pkgutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from fingerprint_engine.handlers.base import FileHandler

from .exceptions import MissingDependencyError, NoHandlerError
from .fft_pipeline import FFTFingerprintPipeline
from .models import Fingerprint, FingerprintConfig, SearchResult

logger = logging.getLogger(__name__)


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
        still align. Applied only under the default config; an explicitly
        customized window_size/hop_size is honored globally instead (so callers
        retain full control and the unit tests' explicit windows are unchanged).
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
                replace(self.config, window_size=window, hop_size=hop)
            )
        return pipelines

    def fingerprint_file(self, path: str | Path) -> Fingerprint:
        """Fingerprint a single file."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        if not source.is_file():
            raise IsADirectoryError(source)

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
        """

        ordered = list(paths)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.fingerprint_file, path) for path in ordered]
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
