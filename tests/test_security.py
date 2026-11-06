"""Cleaning and chunking."""
from src.chunking import (
    CHUNK_SIZE,
    Chunk,
    chunk_document,
    clean_page_text,
    reliability_label,
    split_text,
    strip_repeated_edges,
)


def test_portal_noise_is_removed_but_eligibility_lines_stay():
    text = "\n".join([
        "10/27/26, 10:18 AM State Scholarship Portal",
        "Skip to Content Screen Reader Downloads RTI",
        "Course Type | All",
        "Family annual income <= Rs. 2.5 lakhs",
        "https://ssp.karnataka.gov.in/?section=schemes 2/6",
        "12",
        "Session IP: 117.250.205.100",
    ])
    cleaned, removed = clean_page_text(text)
    assert "Course Type | All" in cleaned
    assert "<= Rs. 2.5 lakhs" in cleaned
    assert "Skip to Content" not in cleaned
    assert "117.250" not in cleaned
    assert "Session IP" not in cleaned
    assert removed >= 3


def test_ip_addresses_inside_a_line_are_removed():
    cleaned, _ = clean_page_text("Contact helpdesk from 192.168.10.4 for support")
    assert "192.168" not in cleaned
    assert "helpdesk" in cleaned


def test_repeated_headers_are_removed_from_long_documents():
    pages = []
    for n in range(1, 8):
        body = [f"Heading of document", *[f"Line {i} of page {n} about topic {i}" for i in range(8)], "Footer text"]
        pages.append("\n".join(body))
    cleaned, removed = strip_repeated_edges(pages)
    assert removed >= 14
    assert all("Heading of document" not in p for p in cleaned)
    assert all("Line 3 of page" in p for p in cleaned)       # middle lines are untouched


def test_short_documents_are_left_alone():
    pages = ["Header\n" + "\n".join(f"line {i}" for i in range(9)) + "\nFooter"] * 3
    cleaned, removed = strip_repeated_edges(pages)
    assert removed == 0 and cleaned == pages


def test_chunks_respect_size_and_lose_no_words():
    words = [f"word{i}" for i in range(900)]
    text = ("\n\n".join(" ".join(words[i:i + 60]) for i in range(0, 900, 60)))
    chunks = split_text(text)
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)


def test_chunks_overlap_between_neighbours():
    text = " ".join(f"w{i}" for i in range(800))
    chunks = split_text(text)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()[-25:]) & set(b.split()[:25])


def test_kannada_text_is_split_safely():
    text = ("ವಿದ್ಯಾರ್ಥಿವೇತನ ಅರ್ಜಿ ಸಲ್ಲಿಕೆ " * 120).strip()
    chunks = split_text(text)
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    assert "ವಿದ್ಯಾರ್ಥಿವೇತನ" in chunks[0]


def test_chunk_ids_are_unique_and_stable(tmp_path):
    f = tmp_path / "ssp_notes.txt"
    f.write_text("\n\n".join(f"Paragraph {i}: " + "scholarship rule " * 20 for i in range(12)), encoding="utf-8")
    first = chunk_document(f, manifest={})
    second = chunk_document(f, manifest={})
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert first[0].chunk_id == "ssp_notes.txt::p0::c1"
    assert first[0].source == "ssp_notes.txt"


def test_pages_stay_separate_in_pdf_chunks(tmp_path):
    from helpers import make_pdf
    f = tmp_path / "guide.pdf"
    f.write_bytes(make_pdf(["First page about income limits and caste certificates.",
                            "Second page about documents required for application."]))
    chunks = chunk_document(f, manifest={})
    assert sorted({c.page for c in chunks}) == [1, 2]
    assert all("Second page" not in c.text for c in chunks if c.page == 1)


def test_reliability_labels():
    assert reliability_label("ssp_dept_social_welfare_schemes.pdf") == "screenshot"
    assert reliability_label("ssp_portal_screenshot_1_overview.pdf") == "screenshot"
    assert reliability_label("aicte_pragati_scholarship_faq.pdf") == "official"


def test_metadata_has_no_empty_values_for_the_database():
    chunk = Chunk("a.pdf::p1::c1", "a.pdf", 1, "text", "native",
                  {"scheme_name": None, "provider": "AICTE"})
    meta = chunk.to_metadata()
    assert None not in meta.values()
    assert meta["provider"] == "AICTE" and meta["scheme_name"] == ""


def test_embedding_text_includes_scheme_context():
    chunk = Chunk("a.pdf::p1::c1", "a.pdf", 1, "Income limit 2.5 lakhs", "native",
                  {"scheme_name": "Fee Reimbursement", "academic_year": "2026-27"})
    assert chunk.embed_text.startswith("Fee Reimbursement | 2026-27")