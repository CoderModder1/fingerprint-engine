"""Fingerprint orchestration and plugin routing."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import pkgutil
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from handlers.base import FileHandler

from .fft_pipeline import FFTFingerprintPipeline
from .models import Fingerprint, FingerprintConfig, SearchResult


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
    ) -> list[Fingerprint]:
        """Fingerprint many files."""


class Fingerprinter(FileProcessor):
    """Routes files through discovered handlers and returns fingerprints."""

    def __init__(
        self,
        config: FingerprintConfig | None = None,
        handlers_package: str = "handlers",
    ) -> None:
        self.config = config or FingerprintConfig()
        self.config.validate()
        self.pipeline = FFTFingerprintPipeline(self.config)
        self.handlers_package = handlers_package
        self.handlers = self.discover_handlers(handlers_package)
        if not self.handlers:
            raise RuntimeError("no file handlers discovered")

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

        for _score, handler in candidates:
            try:
                payload = handler.load(source)
                signal = handler.to_signal(payload)
                landmarks, hashes = handler.extract_peaks(signal, self.pipeline)
                effective_window, effective_hop = self.pipeline.effective_params(signal)
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
            except Exception as exc:
                errors.append(f"{handler.name}: {exc}")

        raise RuntimeError(f"no handler could fingerprint {source}: {'; '.join(errors)}")

    def fingerprint_many(
        self,
        paths: Iterable[str | Path],
        max_workers: int | None = None,
    ) -> list[Fingerprint]:
        """Fingerprint a batch concurrently while preserving input order."""

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.fingerprint_file, paths))

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
