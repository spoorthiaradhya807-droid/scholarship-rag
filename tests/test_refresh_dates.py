"""Reading the last dates from the portal page (no internet needed)."""
from datetime import date

import pytest
from src.refresh_dates import (
    RefreshError,
    academic_year,
    build_document,
    changes,
    check,
    parse_blocks,
    to_json,
)

POST = [
    ("Social Welfare Department", "31/01/2027"),
    ("Tribal Welfare Department", "31/12/2026"),
    ("Backward Classes Welfare Department", "30/11/2026"),
]
PRE = [
    ("Social Welfare Department", "31/12/2026"),
    ("Tribal Welfare Department", "31/12/2026"),
    ("Backward Classes Welfare Department", "30/09/2026"),
]
TODAY = date(2026, 11, 2)


def page(*groups):
    lines = ["Last Dates"]
    for group in groups:
        for department, when in group:
            lines.append(f"Last date to Submit Application for {department} Scheme is: {when} .")
    return "\n".join(lines)


def test_both_lists_are_read_in_order():
    blocks = parse_blocks(page(POST, PRE))
    assert [len(b) for b in blocks] == [3, 3]
    assert blocks[0]["Social Welfare Department"] == date(2027, 1, 31)
    assert blocks[1]["Backward Classes Welfare Department"] == date(2026, 9, 30)
    check(blocks, TODAY)


def test_page_without_dates_is_refused():
    with pytest.raises(RefreshError, match="layout"):
        check(parse_blocks("No dates here"), TODAY)


def test_repeated_page_content_is_refused():
    with pytest.raises(RefreshError, match="layout"):
        check(parse_blocks(page(POST, PRE, POST, PRE)), TODAY)


def test_impossible_dates_are_refused():
    with pytest.raises(RefreshError, match="Unreadable"):
        parse_blocks(page([("Tribal Welfare Department", "99/99/2026")], PRE))
    blocks = parse_blocks(page([("Tribal Welfare Department", "31/12/2035")], PRE))
    with pytest.raises(RefreshError, match="Implausible"):
        check(blocks, TODAY)


def test_document_names_the_category_and_the_collection_date():
    text = build_document(parse_blocks(page(POST, PRE)), TODAY)
    assert "02 November 2026" in text
    assert "Tribal Welfare Department (ST students): last date to submit application is 31/12/2026." in text
    assert "POST-MATRIC" in text and "PRE-MATRIC" in text


def test_extended_deadlines_are_reported():
    old = to_json(parse_blocks(page(
        [("Backward Classes Welfare Department", "31/10/2026")], PRE)))
    new = to_json(parse_blocks(page(
        [("Backward Classes Welfare Department", "30/11/2026")], PRE)))
    found = changes(old, new)
    assert any("2026-10-31 -> 2026-11-30" in line for line in found)
    

def test_academic_year_follows_the_calendar():
    assert academic_year(date(2026, 11, 6)) == "2026-27"
    assert academic_year(date(2027, 2, 1)) == "2026-27"
    assert academic_year(date(2027, 6, 1)) == "2027-28"
    assert academic_year(date(2099, 8, 1)) == "2099-00"