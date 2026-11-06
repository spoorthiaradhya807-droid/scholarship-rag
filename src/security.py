"""Checks for uploaded files."""
import re
from pathlib import Path

from src import config
from src.loaders import SUPPORTED_EXTENSIONS


class UploadError(Exception):
    """The upload was rejected. The message is safe to show to users."""


def sanitize_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name          # drop any folder part
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not stem:
        raise UploadError(
            "The file name is not usable. Rename it using English letters and numbers."
        )
    return stem[:100] + suffix


def validate_upload(name: str, data: bytes) -> str:
    """Return a safe file name, or raise UploadError."""
    safe = sanitize_filename(name)
    suffix = Path(safe).suffix
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadError("Unsupported file type. Upload a PDF, DOCX or TXT file.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError(f"The file is larger than {config.MAX_UPLOAD_MB} MB.")
    if suffix == ".pdf" and not data.lstrip()[:5] == b"%PDF-":
        raise UploadError("This does not look like a real PDF file.")
    if suffix == ".docx" and not data.startswith(b"PK"):
        raise UploadError("This does not look like a real Word (.docx) file.")
    if suffix == ".txt" and b"\x00" in data[:2048]:
        raise UploadError("This does not look like a plain text file.")
    return safe