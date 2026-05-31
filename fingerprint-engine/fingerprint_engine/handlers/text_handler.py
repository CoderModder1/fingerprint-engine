"""Handler for plain text, markup, and source code files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import FileHandler


class TextFileHandler(FileHandler):
    name = "text"
    priority = 40
    default_signal_window = 512
    default_signal_hop = 128
    supported_mime_prefixes = {"text/"}
    supported_mime_types = {
        "application/json",
        "application/javascript",
        "application/xml",
        "application/x-sh",
    }
    supported_extensions = {
        ".txt",
        ".md",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".html",
        ".css",
        ".scss",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".csv",
        ".tsv",
        ".xml",
        ".sh",
        ".sql",
    }

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        base_score = super().can_handle(path, mime_type, sample)
        if base_score:
            return base_score + 0.10
        if not sample:
            return 0.0
        if b"\x00" in sample:
            return 0.0
        try:
            text = sample.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = sample.decode("latin-1")
            except UnicodeDecodeError:
                return 0.0
        if not text:
            return 0.0
        # Score printability on the DECODED code points, not against the
        # ASCII-only ``string.printable``. The old ASCII test rejected
        # accent-dense latin-1 prose (>10% bytes >= 0x80) to the binary handler
        # even though it is perfectly good text. ``str.isprintable()`` accepts
        # any printable Unicode scalar and rejects C0/C1 control characters;
        # ``isspace()`` keeps newlines/tabs (which isprintable() reports False
        # for) counted as legitimate text.
        printable = sum(1 for char in text if char.isprintable() or char.isspace())
        ratio = printable / len(text)
        return 0.55 if ratio >= 0.90 else 0.0

    def load(self, path: str | Path, *, content: bytes | None = None) -> str:
        data = self.read_content(path, content)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")

    def to_signal(self, payload: str) -> np.ndarray:
        if not payload:
            return np.zeros(1, dtype=np.float32)

        code_signal = np.fromiter(
            ((ord(char) % 256) for char in payload),
            dtype=np.float32,
            count=len(payload),
        )
        class_signal = np.fromiter(
            (self._char_class(char) for char in payload),
            dtype=np.float32,
            count=len(payload),
        )
        token_signal = np.asarray(self._token_lengths(payload), dtype=np.float32)

        code_signal = (code_signal - 127.5) / 127.5
        class_signal = (class_signal - 1.5) / 1.5
        if token_signal.max(initial=0.0) > 0:
            token_signal = token_signal / max(1.0, float(token_signal.max()))

        return (0.62 * code_signal) + (0.23 * class_signal) + (0.15 * token_signal)

    def metadata(self, payload: str) -> dict[str, object]:
        return {
            "character_count": len(payload),
            "line_count": payload.count("\n") + (1 if payload else 0),
            "signal_strategy": "char_code_class_token_length",
        }

    @staticmethod
    def _char_class(char: str) -> int:
        if char.isspace():
            return 0
        if char.isalpha():
            return 1
        if char.isdigit():
            return 2
        return 3

    @staticmethod
    def _token_lengths(text: str) -> list[int]:
        lengths: list[int] = []
        current = 0
        for char in text:
            if char.isalnum() or char == "_":
                current += 1
                lengths.append(current)
            else:
                current = 0
                lengths.append(0)
        return lengths
