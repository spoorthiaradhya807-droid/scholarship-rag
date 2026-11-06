"""Official website links."""
import pytest

from src import config
from src.links import official_link


@pytest.fixture(autouse=True)
def no_real_manifest(tmp_path, monkeypatch):
    """Use an empty manifest so your own source_url entries never change these tests."""
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "no_manifest.csv")


def test_ssp_scheme_documents_link_to_the_scheme_details_page():
    _, url = official_link("ssp_dept_social_welfare_schemes.pdf")
    assert url == "https://ssp.karnataka.gov.in/?section=schemes"
    assert official_link("ssp_scholarship_catalogue.txt")[1] == url


def test_ssp_guides_link_to_the_how_to_apply_page():
    assert official_link("ssp_postmatric_documents_required.pdf")[1].endswith("section=process")


def test_central_schemes_link_to_the_right_official_pages():
    assert "aicte-india.org" in official_link("aicte_pragati_scholarship_faq.pdf")[1]
    assert "education.gov.in" in official_link("nsp_pm_usp_csss_faq.pdf")[1]
    assert official_link("nsp_ishan_uday_scholarship_faq.pdf")[1] == "https://scholarships.gov.in/"


def test_unknown_documents_have_no_link():
    assert official_link("sahyadri_science_college_scholarship_policy.pdf") == ("", "")


def test_exact_address_in_the_manifest_wins(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,source_url\naicte_pragati_scholarship_faq.pdf,https://example.gov.in/pragati\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest)
    label, url = official_link("aicte_pragati_scholarship_faq.pdf")
    assert url == "https://example.gov.in/pragati"
    assert label == "example.gov.in"


def test_unsafe_manifest_addresses_are_ignored(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filename,source_url\nnsp_x.pdf,javascript:alert(1)\n", encoding="utf-8")
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest)
    assert official_link("nsp_x.pdf")[1] == "https://scholarships.gov.in/"