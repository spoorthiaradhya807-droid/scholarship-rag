"""Document loading: PDF, DOCX, TXT, OCR text, manual fixes and errors."""
import json

import pytest
from docx import Document

from helpers import make_pdf
from src import config, loaders
from src.loaders import DocumentLoadError, load_document

LONG_TEXT = "Eligibility: family annual income up to Rs. 2.5 lakhs for all courses."


def test_txt_loads_as_one_page(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("First line\x00 with a null character", encoding="utf-8")
    pages = load_document(f, manifest={})
    assert len(pages) == 1
    assert pages[0].page is None
    assert pages[0].text_source == "txt"
    assert "\x00" not in pages[0].text


def test_unsupported_extension_is_rejected(tmp_path):
    f = tmp_path / "program.exe"
    f.write_bytes(b"MZ")
    with pytest.raises(DocumentLoadError, match="unsupported"):
        load_document(f, manifest={})


def test_empty_text_file_is_rejected(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="no readable text"):
        load_document(f, manifest={})


def test_pdf_pages_keep_their_numbers(tmp_path):
    f = tmp_path / "guide.pdf"
    f.write_bytes(make_pdf([LONG_TEXT, "Second page lists the documents required."]))
    pages = load_document(f, manifest={})
    assert [p.page for p in pages] == [1, 2]
    assert all(p.text_source == "native" for p in pages)
    assert "2.5 lakhs" in pages[0].text


def test_corrupt_pdf_gives_a_friendly_error(tmp_path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"this is not a pdf at all")
    with pytest.raises(DocumentLoadError):
        load_document(f, manifest={})


def test_scanned_pdf_without_ocr_is_reported(tmp_path):
    f = tmp_path / "scan.pdf"
    f.write_bytes(make_pdf([""]))
    with pytest.raises(DocumentLoadError, match="scanned"):
        load_document(f, manifest={})


def test_saved_ocr_text_is_used(tmp_path, monkeypatch):
    f = tmp_path / "scan.pdf"
    f.write_bytes(make_pdf([""]))
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "scan.json").write_text(
        json.dumps({"pages": {"1": "Scheme: Fee Reimbursement"}}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "OCR_DIR", ocr_dir)
    pages = load_document(f, manifest={})
    assert pages[0].text == "Scheme: Fee Reimbursement"
    assert pages[0].text_source == "ocr"


def test_manual_fix_beats_other_sources(tmp_path, monkeypatch):
    f = tmp_path / "guide.pdf"
    f.write_bytes(make_pdf(["Old extracted text for page one", "Page two text stays"]))
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "guide_p1.txt").write_text("Corrected text", encoding="utf-8")
    monkeypatch.setattr(loaders, "MANUAL_DIR", manual)
    pages = load_document(f, manifest={})
    assert pages[0].text == "Corrected text" and pages[0].text_source == "manual"
    assert pages[1].text_source == "native"


def test_empty_manual_file_removes_a_page(tmp_path, monkeypatch):
    f = tmp_path / "guide.pdf"
    f.write_bytes(make_pdf(["Outdated dates on page one", "Useful content on page two"]))
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "guide_p1.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(loaders, "MANUAL_DIR", manual)
    pages = load_document(f, manifest={})
    assert [p.page for p in pages] == [2]


def test_docx_text_and_tables_are_read(tmp_path):
    f = tmp_path / "rules.docx"
    doc = Document()
    doc.add_paragraph("Scholarship rules")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Income limit"
    table.rows[0].cells[1].text = "2.5 lakhs"
    doc.save(str(f))
    pages = load_document(f, manifest={})
    assert "Scholarship rules" in pages[0].text
    assert "Income limit | 2.5 lakhs" in pages[0].text
    assert pages[0].text_source == "docx"


def test_manifest_details_are_attached(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Some scholarship text", encoding="utf-8")
    manifest = {"notes.txt": {"scheme_name": "Test Scheme", "academic_year": "2026-27"}}
    page = load_document(f, manifest)[0]
    assert page.metadata["scheme_name"] == "Test Scheme"
    assert page.metadata["academic_year"] == "2026-27"
    assert page.metadata["provider"] == ""