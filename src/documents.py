"""Admin helpers: add, replace and remove documents, keeping the database in sync."""
import csv
import shutil
from datetime import date

from src import config
from src.security import validate_upload
from src.vectorstore import index_document, remove_document

UPLOAD_DIR = config.RAW_DIR / "uploaded"
REMOVED_DIR = config.BASE_DIR / "data" / "removed"
MANIFEST_COLUMNS = [
    "filename", "original_filename", "scheme_name", "provider", "category",
    "academic_year", "language", "pages", "text_layer", "source_url",
    "date_collected", "notes",
]


def find_file(name: str):
    for path in config.RAW_DIR.rglob("*"):
        if path.is_file() and path.name == name:
            return path
    return None


def update_manifest(filename: str, fields: dict):
    """Add or update the manifest row for one file."""
    columns, rows = list(MANIFEST_COLUMNS), []
    if config.MANIFEST_PATH.exists():
        with open(config.MANIFEST_PATH, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or columns)
            rows = list(reader)
    for row in rows:
        if row.get("filename") == filename:
            row.update(fields)
            break
    else:
        row = {c: "" for c in columns}
        row["filename"] = filename
        row.update(fields)
        rows.append(row)
    with open(config.MANIFEST_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_document(filename: str, data: bytes, meta: dict = None):
    """Validate, save and index an uploaded file (replaces a file with the same name).
    Returns (saved name, status). Rolls the file back if indexing fails."""
    safe = validate_upload(filename, data)
    existing = find_file(safe)
    target = existing or (UPLOAD_DIR / safe)
    old_bytes = target.read_bytes() if existing else None

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    old_manifest = (
        config.MANIFEST_PATH.read_bytes() if config.MANIFEST_PATH.exists() else None
    )
    target.write_bytes(data)
    try:
        fields = {k: v for k, v in (meta or {}).items() if v}
        fields.update(original_filename=filename, date_collected=date.today().isoformat())
        update_manifest(safe, fields)
        status = index_document(target)
    except Exception:
        if old_manifest is None:
            config.MANIFEST_PATH.unlink(missing_ok=True)
        else:
            config.MANIFEST_PATH.write_bytes(old_manifest)
        if old_bytes is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(old_bytes)
        raise
    return safe, status


def delete_document(name: str) -> int:
    """Remove a document from the database and move its file to data/removed."""
    removed = remove_document(name)
    path = find_file(name)
    if path:
        REMOVED_DIR.mkdir(parents=True, exist_ok=True)
        destination = REMOVED_DIR / name
        destination.unlink(missing_ok=True)
        shutil.move(str(path), str(destination))
    return removed