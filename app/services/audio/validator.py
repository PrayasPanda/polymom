"""Upload validation: extension, magic bytes, size limit, ffprobe audio check.

Uploads are streamed to disk in chunks (never fully buffered in memory), checked,
then atomically renamed to ``{stem}.{ext}``. Any failure removes partial files.
"""

import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import filetype
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import (
    AudioTooLongError,
    CorruptedMediaError,
    EmptyFileError,
    FileTooLargeError,
    MediaProbeUnavailableError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.schemas.meeting import AudioMetadata

logger = get_logger(__name__)

# Bytes needed by ``filetype`` to recognise every container we accept.
MAGIC_HEAD_BYTES = 8192
MAX_FILENAME_LEN = 255

_ISO_BMFF = frozenset({"audio/mp4", "audio/x-m4a", "video/mp4", "video/quicktime", "video/3gpp"})
_MATROSKA = frozenset({"video/x-matroska", "video/webm"})

# Detected MIME types accepted for each extension. Containers sharing a format
# (m4a/mp4/mov, mkv/webm) are interchangeable at the byte level.
EXTENSION_MIME_TYPES: dict[str, frozenset[str]] = {
    "wav": frozenset({"audio/x-wav", "audio/wav"}),
    "mp3": frozenset({"audio/mpeg"}),
    "flac": frozenset({"audio/x-flac", "audio/flac"}),
    "ogg": frozenset({"audio/ogg", "video/ogg"}),
    "aac": frozenset({"audio/aac"}),
    "m4a": _ISO_BMFF,
    "mp4": _ISO_BMFF,
    "mov": _ISO_BMFF,
    "mkv": _MATROSKA,
    "webm": _MATROSKA,
}


class AsyncReadable(Protocol):
    """Anything with an async ``read`` (e.g. ``fastapi.UploadFile``)."""

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ValidatedMedia:
    path: Path
    original_filename: str
    extension: str
    mime_type: str
    size_bytes: int
    metadata: AudioMetadata
    sha256: str


def _is_name_char(char: str) -> bool:
    """Letters (L*), combining marks (M*) and numbers (N*)."""
    return unicodedata.category(char)[0] in "LMN"


def sanitize_filename(name: str | None) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips directories, control and shell-special characters and leading dots,
    keeps Unicode letters, combining marks and digits (so Devanagari/Odia names
    with vowel signs survive), and caps the length.
    """
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c if c in "._- " or _is_name_char(c) else "_" for c in name)
    name = re.sub(r"_{2,}", "_", name).strip(" .")
    if len(name) > MAX_FILENAME_LEN:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) < 16:
            name = f"{stem[: MAX_FILENAME_LEN - len(ext) - 1]}.{ext}"
        else:
            name = name[:MAX_FILENAME_LEN]
    return name or "upload"


class MediaValidator:
    """Validates and stores uploaded media files."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def check_extension(self, filename: str) -> str:
        """Return the lower-case extension, or raise if it is not allowed."""
        ext = Path(filename).suffix.lower().lstrip(".")
        allowed = self._settings.allowed_extensions & EXTENSION_MIME_TYPES.keys()
        if ext not in allowed:
            raise UnsupportedFileTypeError(
                f"File extension '.{ext}' is not allowed." if ext else "File has no extension.",
                details={"extension": ext or None, "allowed": sorted(allowed)},
            )
        return ext

    @staticmethod
    def check_magic(head: bytes, ext: str) -> str:
        """Detect the real MIME type from magic bytes and match it to ``ext``."""
        kind = filetype.guess(head)
        detected: str | None = kind.mime if kind is not None else None
        if detected is None or detected not in EXTENSION_MIME_TYPES[ext]:
            raise UnsupportedFileTypeError(
                "File content does not match its extension.",
                details={"extension": ext, "detected_mime_type": detected},
            )
        return detected

    async def stream_to_disk(self, source: AsyncReadable, dest: Path) -> tuple[int, bytes, str]:
        """Copy ``source`` to ``dest`` chunk by chunk, enforcing the size limit.

        Returns the byte count, the leading bytes needed for magic detection and the
        SHA-256 hex digest (used to detect duplicate uploads).
        """
        limit = self._settings.max_upload_bytes
        size = 0
        head = bytearray()
        digest = hashlib.sha256()
        with dest.open("xb") as out:
            while chunk := await source.read(self._settings.upload_chunk_bytes):
                size += len(chunk)
                if size > limit:
                    raise FileTooLargeError(
                        f"File exceeds the {self._settings.max_upload_mb} MB upload limit.",
                        details={"max_upload_mb": self._settings.max_upload_mb},
                    )
                if len(head) < MAGIC_HEAD_BYTES:
                    head += chunk[: MAGIC_HEAD_BYTES - len(head)]
                digest.update(chunk)
                await run_in_threadpool(out.write, chunk)
        return size, bytes(head), digest.hexdigest()

    async def probe(self, path: Path) -> AudioMetadata:
        """Run ffprobe and return metadata for the first audio stream."""
        cmd = [
            self._settings.ffprobe_path,
            *("-v", "error", "-print_format", "json", "-show_format", "-show_streams"),
            # Prevent SSRF / LFI via crafted containers that reference external URLs.
            *("-protocol_whitelist", "file,pipe,fd"),
            str(path),
        ]
        try:
            result = await run_in_threadpool(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=self._settings.ffprobe_timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MediaProbeUnavailableError(
                "Media probing is unavailable (ffprobe not found)."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CorruptedMediaError(
                "Timed out while reading the media file.", details={"reason": "probe_timeout"}
            ) from exc

        if result.returncode != 0:
            logger.info("ffprobe_failed", returncode=result.returncode, stderr=result.stderr[-500:])
            raise CorruptedMediaError(
                "The media file is corrupted or unreadable.", details={"reason": "unreadable"}
            )
        try:
            info: dict[str, Any] = json.loads(result.stdout or b"{}")
        except json.JSONDecodeError as exc:
            raise CorruptedMediaError(
                "The media file is corrupted or unreadable.", details={"reason": "unreadable"}
            ) from exc

        return self._audio_metadata(info)

    def _check_duration(self, metadata: AudioMetadata) -> None:
        limit_seconds = self._settings.max_audio_duration_minutes * 60
        duration = metadata.duration_seconds or 0.0
        if duration > limit_seconds:
            raise AudioTooLongError(
                f"Audio is {duration / 60:.1f} minutes; the limit is "
                f"{self._settings.max_audio_duration_minutes:g} minutes.",
                details={
                    "duration_seconds": round(duration, 1),
                    "max_audio_duration_minutes": self._settings.max_audio_duration_minutes,
                },
            )

    @staticmethod
    def _audio_metadata(info: dict[str, Any]) -> AudioMetadata:
        streams: list[dict[str, Any]] = info.get("streams") or []
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if audio is None:
            raise CorruptedMediaError(
                "The file contains no audio stream.", details={"reason": "no_audio_stream"}
            )
        fmt: dict[str, Any] = info.get("format") or {}
        duration = _to_float(audio.get("duration")) or _to_float(fmt.get("duration"))
        if not duration or duration <= 0:
            raise CorruptedMediaError(
                "The audio stream is empty.", details={"reason": "zero_duration"}
            )
        return AudioMetadata(
            duration_seconds=round(duration, 3),
            codec=audio.get("codec_name"),
            sample_rate=_to_int(audio.get("sample_rate")),
            channels=_to_int(audio.get("channels")),
            format_name=fmt.get("format_name"),
            bit_rate=_to_int(audio.get("bit_rate")) or _to_int(fmt.get("bit_rate")),
        )

    async def validate(
        self, source: AsyncReadable, filename: str | None, dest_dir: Path, stem: str
    ) -> ValidatedMedia:
        """Validate an upload and store it as ``dest_dir/{stem}.{ext}``."""
        original = sanitize_filename(filename)
        ext = self.check_extension(original)
        await run_in_threadpool(dest_dir.mkdir, parents=True, exist_ok=True)
        partial = dest_dir / f"{stem}.{ext}.part"
        final = dest_dir / f"{stem}.{ext}"
        try:
            size, head, sha256 = await self.stream_to_disk(source, partial)
            if size == 0:
                raise EmptyFileError("The uploaded file is empty.")
            mime = self.check_magic(head, ext)
            metadata = await self.probe(partial)
            self._check_duration(metadata)
            partial.replace(final)
        except BaseException:
            partial.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise
        return ValidatedMedia(
            path=final,
            original_filename=original,
            extension=ext,
            mime_type=mime,
            size_bytes=size,
            metadata=metadata,
            sha256=sha256,
        )


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
