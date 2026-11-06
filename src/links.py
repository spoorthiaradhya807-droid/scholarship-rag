"""Official website links shown under each answer."""
import csv
from urllib.parse import urlparse

from src import config
SSP_SCHEMES = "https://ssp.karnataka.gov.in/?section=schemes"
SSP_PROCESS = "https://ssp.karnataka.gov.in/?section=process"
SCHEMES_LABEL = "Karnataka SSP: Scheme details (choose post-matric or pre-matric, then the department and the scheme)"
PROCESS_LABEL = "Karnataka SSP: How to apply"

# (file name starts with, link text, address). The first match wins.
PORTAL_RULES = [
    ("ssp_dept_", SCHEMES_LABEL, SSP_SCHEMES),
    ("ssp_scholarship_catalogue", SCHEMES_LABEL, SSP_SCHEMES),
    ("ssp_arya_vysya_", SCHEMES_LABEL, SSP_SCHEMES),
    ("ssp_postmatric_scheme_", SCHEMES_LABEL, SSP_SCHEMES),
    ("ssp_prematric_scheme_", SCHEMES_LABEL, SSP_SCHEMES),
    ("ssp_portal_screenshot_", "Karnataka State Scholarship Portal (SSP)", "https://ssp.karnataka.gov.in/"),
    ("ssp_postmatric_", PROCESS_LABEL, SSP_PROCESS),
    ("ssp_prematric_", PROCESS_LABEL, SSP_PROCESS),
    ("ssp_", "Karnataka State Scholarship Portal (SSP)", "https://ssp.karnataka.gov.in/"),
    ("aicte_", "AICTE student development schemes (Pragati, Saksham, Swanath)",
     "https://www.aicte-india.org/schemes/students-development-schemes"),
    ("nsp_pm_usp", "PM-USP Central Sector Scheme (Ministry of Education)",
     "https://www.education.gov.in/node/255"),
    ("nsp_", "National Scholarship Portal (NSP)", "https://scholarships.gov.in/"),
]


def _manifest_urls() -> dict:
    """File name -> exact web address, for rows where you filled source_url."""
    if not config.MANIFEST_PATH.exists():
        return {}
    urls = {}
    with open(config.MANIFEST_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            url = (row.get("source_url") or "").strip()
            if url.lower().startswith(("https://", "http://")):
                urls[row.get("filename", "")] = url
    return urls


def official_link(filename: str):
    """Return (link text, address) for a document, or ("", "") if none is known."""
    url = _manifest_urls().get(filename)
    if url:
        return urlparse(url).netloc, url
    for prefix, label, address in PORTAL_RULES:
        if filename.startswith(prefix):
            return label, address
    return "", ""