"""Load documents (PDF, DOCX, TXT) as page-level text with metadata.

For each PDF page the text comes from, in this order:
1. a manual correction file: data/processed/manual/<pdf name>_p<page>.txt
   (an EMPTY file is deliberate: it blanks the page so it is not indexed)
2. the Gemini OCR text saved by ocr_scanned.py
3. the PDF's own text layer
"""
import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from docx import Document as DocxDocument
from pypdf import PdfReader

from src import config

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MANUAL_DIR = config.BASE_DIR / "data" / "processed" / "manual"
MANIFEST_FIELDS = (
    "scheme_name", "provider", "category", "academic_year", "source_url", "language",
)
KANNADA = re.compile(r"[\u0C80-\u0CFF]")
GARBLED = re.compile(r"[\uE000-\uF8FF\uFFFD]")


class DocumentLoadError(Exception):
    """A document could not be read. The message is safe to show to users."""


@dataclass
class PageText:
    source: str                 # file name, shown in citations
    page: Optional[int]         # 1-based page number (None for DOCX/TXT)
    text: str
    text_source: str            # "native", "ocr", "manual", "docx" or "txt"
    metadata: dict = field(default_factory=dict)


def clean_basic(text: str) -> str:
    """Remove null characters and normalise Unicode (helps Kannada text)."""
    return unicodedata.normalize("NFC", text.replace("\x00", "")).strip()


def load_manifest() -> dict:
    """Read data/manifest.csv into {file name: row}."""
    if not config.MANIFEST_PATH.exists():
        return {}
    with open(config.MANIFEST_PATH, newline="", encoding="utf-8-sig") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def manifest_metadata(filename: str, manifest: dict) -> dict:
    row = manifest.get(filename, {})
    return {key: (row.get(key) or "") for key in MANIFEST_FIELDS}


def _read_ocr_pages(path: Path) -> dict:
    ocr_file = config.OCR_DIR / f"{path.stem}.json"
    if not ocr_file.exists():
        return {}
    return json.loads(ocr_file.read_text(encoding="utf-8")).get("pages", {})


def _choose_pdf_text(path: Path, number: int, page, ocr_pages: dict):
    manual_file = MANUAL_DIR / f"{path.stem}_p{number}.txt"
    if manual_file.exists():
        return clean_basic(manual_file.read_text(encoding="utf-8")), "manual"

    ocr_text = clean_basic(ocr_pages.get(str(number), ""))
    if ocr_text:
        return ocr_text, "ocr"

    try:
        native = clean_basic(page.extract_text() or "")
    except Exception:
        native = ""
    return native, "native"

def load_pdf(path: Path) -> list:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise DocumentLoadError(f"{path.name} is password-protected.")
        count = len(reader.pages)
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"{path.name} could not be read as a PDF ({exc}).")

    ocr_pages = _read_ocr_pages(path)
    pages = []
    for number in range(1, count + 1):
        text, origin = _choose_pdf_text(path, number, reader.pages[number - 1], ocr_pages)
        pages.append(PageText(path.name, number, text, origin))
    return pages


def load_docx(path: Path) -> list:
    try:
        doc = DocxDocument(str(path))
    except Exception as exc:
        raise DocumentLoadError(f"{path.name} could not be read as a Word file ({exc}).")
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [PageText(path.name, None, clean_basic("\n".join(parts)), "docx")]


def load_txt(path: Path) -> list:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [PageText(path.name, None, clean_basic(text), "txt")]


def load_document(path, manifest: dict = None, include_empty: bool = False) -> list:
    """Load one file as a list of PageText, with manifest metadata attached."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"{path.name}: unsupported file type. Use PDF, DOCX or TXT."
        )
    loader = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt}[suffix]
    pages = loader(path)

    metadata = manifest_metadata(path.name, manifest if manifest is not None else load_manifest())
    for page in pages:
        page.metadata = dict(metadata)

    kept = pages if include_empty else [p for p in pages if p.text]
    if not kept:
        raise DocumentLoadError(
            f"{path.name}: no readable text found. It may be a scanned document."
        )
    return kept


def looks_garbled(text: str) -> bool:
    if not text:
        return False
    bad = len(GARBLED.findall(text)) + 5 * text.count("(cid:")
    return bad / len(text) > 0.02


def main():
    manifest = load_manifest()
    files = sorted(
        p for p in config.RAW_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    lines, origins = [], Counter()
    total_pages = 0
    for path in files:
        try:
            pages = load_document(path, manifest, include_empty=True)
        except DocumentLoadError as exc:
            lines.append(f"{path.name}: ERROR {exc}")
            continue
        origin = Counter(p.text_source for p in pages if p.text)
        origins.update(origin)
        total_pages += len(pages)
        empty = [p.page for p in pages if not p.text]
        garbled = [p.page for p in pages if looks_garbled(p.text)]
        kannada = sum(len(KANNADA.findall(p.text)) for p in pages)
        lines.append(
            f"{path.name} | pages={len(pages)} native={origin['native']} "
            f"ocr={origin['ocr']} manual={origin['manual']} | "
            f"empty={empty or '-'} garbled={garbled or '-'} kannada_chars={kannada}"
        )
    lines.append("")
    lines.append(f"TOTAL: {len(files)} files, {total_pages} pages, text from {dict(origins)}")

    report = "\n".join(lines)
    out = config.BASE_DIR / "data" / "processed" / "load_report.txt"
    out.write_text(report, encoding="utf-8")
    print(report.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()