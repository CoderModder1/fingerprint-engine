from __future__ import annotations

import importlib
import sys
from io import BytesIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.handlers.audio_handler import AudioFileHandler
from fingerprint_engine.handlers.base import FileHandler
from fingerprint_engine.handlers.image_handler import ImageFileHandler
from fingerprint_engine.handlers.pdf_handler import PDFFileHandler
from fingerprint_engine.handlers.text_handler import TextFileHandler


def test_pdf_handler_missing_pypdf_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A missing pypdf must fail loud with MissingDependencyError rather than be
    # silently swallowed into the latin1 byte fallback, which would produce a
    # divergent fingerprint incomparable to one made where pypdf is installed
    # (silent index corruption).
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\nsome real-looking pdf content\n%%EOF\n")

    # Simulate pypdf being absent by making the import_module call require_optional
    # uses raise (robust regardless of whether pypdf's submodules are cached).
    real_import_module = importlib.import_module

    def _fake_import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("No module named 'pypdf'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    with pytest.raises(MissingDependencyError) as excinfo:
        PDFFileHandler().load(path)

    assert excinfo.value.package == "pypdf"
    assert excinfo.value.extra == "pdf"


def test_pdf_handler_corrupt_pdf_still_degrades_to_latin1(tmp_path: Path) -> None:
    # A genuine parse failure on a corrupt/unparseable PDF (pypdf installed)
    # must still degrade gracefully to the latin1 byte fallback rather than
    # raise -- only the missing-dependency case fails loud.
    pytest.importorskip("pypdf", exc_type=ImportError)

    path = tmp_path / "garbage.pdf"
    path.write_bytes(b"%PDF-1.4 garbage")

    payload = PDFFileHandler().load(path)

    assert payload.parser == "latin1_fallback"
    assert payload.page_count == 0


# ---------------------------------------------------------------------------
# base.can_handle: strongest evidence (MAX score) wins
# ---------------------------------------------------------------------------


class _MaxScoreHandler(FileHandler):
    """Handler whose extension AND exact MIME both match the same file."""

    name = "maxscore"
    supported_extensions = {".foo"}
    supported_mime_types = {"application/x-foo"}

    def load(self, path: str | Path) -> object:  # pragma: no cover - unused
        raise NotImplementedError

    def to_signal(self, payload: object):  # pragma: no cover - unused
        raise NotImplementedError


def test_can_handle_returns_max_score_not_first_match() -> None:
    # Both the extension (0.75) and the exact MIME (0.80) match. The old code
    # returned the extension score first and ignored the higher MIME score; the
    # fix must return the MAX so the stronger MIME evidence wins.
    score = _MaxScoreHandler.can_handle(
        "doc.foo", mime_type="application/x-foo", sample=None
    )
    assert score == pytest.approx(0.80)


def test_can_handle_extension_only_still_scores() -> None:
    # Extension matches, MIME does not -> extension score, not 0.
    assert _MaxScoreHandler.can_handle("doc.foo", mime_type=None) == pytest.approx(0.75)


def test_can_handle_mime_only_still_scores() -> None:
    # Exact MIME matches, extension does not -> MIME score, not 0.
    score = _MaxScoreHandler.can_handle("doc.bar", mime_type="application/x-foo")
    assert score == pytest.approx(0.80)


def test_can_handle_no_match_is_zero() -> None:
    assert _MaxScoreHandler.can_handle("doc.bar", mime_type="text/plain") == 0.0


# ---------------------------------------------------------------------------
# Audio routing: non-WAV/MP3 audio falls through; no force-decode-as-mp3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".ogg", ".flac", ".m4a", ".aac", ".opus"])
def test_audio_handler_excludes_unsupported_audio_formats(ext: str) -> None:
    # .ogg/.flac/.m4a/.aac used to route here via the broad ``audio/`` MIME
    # prefix and then get force-decoded as MP3 (garbage). They must now score
    # 0.0 so they fall through to text/binary deliberately.
    mime = {
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".opus": "audio/opus",
    }[ext]
    score = AudioFileHandler.can_handle(f"clip{ext}", mime_type=mime, sample=None)
    assert score == 0.0


@pytest.mark.parametrize(
    ("name", "mime"),
    [
        ("clip.wav", "audio/wav"),
        ("clip.mp3", "audio/mpeg"),
    ],
)
def test_audio_handler_still_accepts_wav_and_mp3(name: str, mime: str) -> None:
    assert AudioFileHandler.can_handle(name, mime_type=mime, sample=None) > 0.0


def test_audio_load_does_not_hardcode_mp3_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The MP3 loader must let ffmpeg sniff the container (no ``format=`` kwarg)
    # rather than force-decode as mp3, which turned any non-mp3 input into
    # garbage. Verify AudioSegment.from_file is called WITHOUT a format kwarg.
    pydub = pytest.importorskip("pydub", exc_type=ImportError)

    class _FakeSegment:
        channels = 1
        sample_width = 2
        frame_rate = 8000

        def get_array_of_samples(self) -> list[int]:
            return [0, 1, -1, 2]

    captured: dict[str, object] = {}

    def _fake_from_file(path: object, *args: object, **kwargs: object) -> _FakeSegment:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeSegment()

    monkeypatch.setattr(pydub.AudioSegment, "from_file", staticmethod(_fake_from_file))

    path = tmp_path / "clip.mp3"
    path.write_bytes(b"ID3 not really mp3")
    # _load_mp3 takes the already-read bytes (single-read contract); the public
    # load() threads them in. from_file must still be called WITHOUT a format kwarg.
    AudioFileHandler()._load_mp3(path.read_bytes())

    assert "format" not in captured["kwargs"]  # type: ignore[operator]
    assert captured["args"] == ()


def test_pydub_is_importable_wherever_installed() -> None:
    # Guards the [audio] extra's `audioop-lts; python_version >= '3.13'` pin.
    # pydub imports the stdlib `audioop`, which PEP 594 removed in Python 3.13, so
    # without the backport `import pydub` raises on 3.13 and MP3 fingerprinting is
    # dead even with [audio] installed. A plain importorskip would MASK that by
    # skipping a broken-but-installed pydub; instead skip ONLY when pydub is not
    # installed at all, and otherwise require the import to actually succeed.
    import importlib.util

    if importlib.util.find_spec("pydub") is None:
        pytest.skip("pydub is not installed")
    importlib.import_module("pydub")  # must not raise where pydub is installed


def test_load_mp3_decodes_a_real_mp3_end_to_end() -> None:
    # End-to-end REAL decode (not the fake-segment unit test above): synthesize a
    # short sine, encode it to genuine MP3 bytes via ffmpeg, then decode through
    # the handler. Skips cleanly where no MP3 *encoder* is available to build the
    # fixture (e.g. ffmpeg/lame absent), so it adds real coverage where it can
    # without being flaky.
    import numpy as np

    pydub = pytest.importorskip("pydub", exc_type=ImportError)

    sample_rate = 8000
    t = np.arange(int(sample_rate * 0.3)) / sample_rate
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()
    segment = pydub.AudioSegment(
        data=pcm, sample_width=2, frame_rate=sample_rate, channels=1
    )

    buffer = BytesIO()
    try:
        segment.export(buffer, format="mp3")
    except Exception as exc:  # no MP3 encoder (ffmpeg/lame) on this box
        pytest.skip(f"no MP3 encoder available to synthesize a fixture: {exc}")
    mp3_bytes = buffer.getvalue()
    if not mp3_bytes:
        pytest.skip("MP3 encoder produced no output")

    payload = AudioFileHandler()._load_mp3(mp3_bytes)

    assert payload.samples.size > 0
    assert payload.sample_rate > 0
    assert payload.decoder == "pydub.ffmpeg"


# ---------------------------------------------------------------------------
# Text sniffer: accent-dense latin-1 prose is accepted, not rejected to binary
# ---------------------------------------------------------------------------


def test_text_handler_accepts_accent_dense_latin1() -> None:
    # French/Spanish prose with >10% non-ASCII bytes. The old ASCII-only
    # ``string.printable`` test scored these chars as non-printable and rejected
    # the file to the binary handler; scoring on decoded code points with
    # isprintable() must now accept it.
    prose = "Très élégant café au lait à la française. ¿Cómo estás? ñ ü ö ä é è ê ç à"
    sample = prose.encode("latin-1")
    assert sum(1 for b in sample if b >= 0x80) / len(sample) > 0.10
    score = TextFileHandler.can_handle("note.txt", mime_type=None, sample=sample)
    assert score >= 0.55


def test_text_handler_rejects_control_char_dense_binary() -> None:
    # C0/C1 control chars are non-printable; a control-dense blob (no NUL) must
    # still be rejected so binary doesn't masquerade as text.
    sample = bytes(range(1, 9)) * 20  # \x01-\x08 repeated, all C0 controls
    score = TextFileHandler.can_handle("blob.bin", mime_type=None, sample=sample)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Image routing: SVG / text-based image MIME does not route to the decoder
# ---------------------------------------------------------------------------


def test_image_handler_excludes_svg() -> None:
    # image/svg+xml shares the ``image/`` prefix but is XML text the raster
    # decoder cannot handle; it must score 0.0 and fall to text/binary. The
    # default-mode raster handler is the one that would claim images, so the
    # decline is asserted on a default (raster) instance.
    score = ImageFileHandler().can_handle(
        "logo.svg", mime_type="image/svg+xml", sample=b"<svg xmlns='...'></svg>"
    )
    assert score == 0.0


def test_image_handler_still_accepts_png() -> None:
    # Default mode is raster, so a fresh raster instance claims a PNG.
    score = ImageFileHandler().can_handle(
        "pic.png", mime_type="image/png", sample=b"\x89PNG\r\n\x1a\n"
    )
    assert score > 0.0


def test_image_handler_sniffs_webp_without_extension_or_mime() -> None:
    # B3: an extension-less / MIME-less WEBP (RIFF....WEBP) must be claimed via
    # the magic-byte sniff, not fall through to the binary handler (which would
    # not match the same WEBP fingerprinted with its extension present).
    webp_sample = b"RIFF\x24\x00\x00\x00WEBPVP8 "
    score = ImageFileHandler().can_handle("noext", mime_type=None, sample=webp_sample)
    assert score == pytest.approx(0.90)
    # A RIFF that is NOT WEBP (e.g. a WAV) is not claimed by the image handler.
    wav_sample = b"RIFF\x24\x00\x00\x00WAVEfmt "
    assert ImageFileHandler().can_handle("noext", mime_type=None, sample=wav_sample) == 0.0


# ---------------------------------------------------------------------------
# PDF page cap + structural-only warning
# ---------------------------------------------------------------------------


def _make_multipage_pdf(num_pages: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_handler_page_cap_limits_extraction(tmp_path: Path) -> None:
    pytest.importorskip("pypdf", exc_type=ImportError)

    path = tmp_path / "many.pdf"
    path.write_bytes(_make_multipage_pdf(5))

    capped = PDFFileHandler(max_pdf_pages=2).load(path)
    assert capped.parser == "pypdf"
    assert capped.page_count == 2
    assert capped.text.count("[[PAGE:") == 2

    # 0 = unlimited: all five pages extracted.
    uncapped = PDFFileHandler(max_pdf_pages=0).load(path)
    assert uncapped.page_count == 5
    assert uncapped.text.count("[[PAGE:") == 5


def test_pdf_handler_default_is_unlimited(tmp_path: Path) -> None:
    pytest.importorskip("pypdf", exc_type=ImportError)

    path = tmp_path / "three.pdf"
    path.write_bytes(_make_multipage_pdf(3))

    payload = PDFFileHandler().load(path)  # no-arg construction -> unlimited
    assert payload.page_count == 3


def test_pdf_handler_rejects_negative_page_cap() -> None:
    with pytest.raises(ValueError, match="max_pdf_pages"):
        PDFFileHandler(max_pdf_pages=-1)


def test_pdf_handler_corrupt_pdf_warns_structural_only(tmp_path: Path) -> None:
    # On a genuine parse failure the latin1 fallback must fire AND a
    # RuntimeWarning must announce that the fingerprint is structural-only, so
    # it is not silently mistaken for real text extraction.
    pytest.importorskip("pypdf", exc_type=ImportError)

    path = tmp_path / "garbage.pdf"
    path.write_bytes(b"%PDF-1.4 garbage")

    with pytest.warns(RuntimeWarning, match="structural-only"):
        payload = PDFFileHandler().load(path)

    assert payload.parser == "latin1_fallback"
    assert payload.page_count == 0
