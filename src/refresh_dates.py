"""Read the last dates from the public SSP home page and keep them as a document.

Run it at most once a day. It makes ONE request to a public page, never logs in,
and keeps the old file if anything looks wrong.
"""
import argparse
import json
import re
from datetime import date, datetime

import httpx

from src import config
from src.fetch_page import visible_text

URL = "https://ssp.karnataka.gov.in/"
USER_AGENT = "ScholarshipRAG-student-project/1.0 (reads public last-date notices once a day)"
OUTPUT = config.RAW_DIR / "karnataka" / "ssp_last_dates_current.txt"
SNAPSHOT = config.BASE_DIR / "data" / "processed" / "live" / "ssp_last_dates.json"
LEVELS = [
    ("post", "POST-MATRIC (Class 11 and above, including PUC)"),
    ("pre", "PRE-MATRIC (Class 1 to 10)"),
]
WHO = {
    "Social Welfare Department": "SC students",
    "Tribal Welfare Department": "ST students",
    "Backward Classes Welfare Department": "OBC, Category-1 and NT/SNT students",
    "Minorities Welfare Department": "minority students",
}
LINE = re.compile(
    r"Last date to Submit Application for (.+?) Scheme is:\s*(\d{1,2}/\d{1,2}/\d{4})", re.I
)


def academic_year(today: date) -> str:
    """Indian academic year label, e.g. 2026-27 (the year starts in June)."""
    start = today.year if today.month >= 6 else today.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


class RefreshError(Exception):
    """Something looked wrong, so the existing file was left untouched."""


def fetch_text() -> str:
    try:
        response = httpx.get(
            URL, timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
    except Exception as exc:
        raise RefreshError(f"Could not download {URL}: {type(exc).__name__}")
    return visible_text(response.text)


def parse_blocks(text: str):
    """Group the 'Last date' lines. A new group starts when a department repeats."""
    blocks, current = [], {}
    for match in LINE.finditer(text):
        department, raw = match.group(1).strip(), match.group(2)
        try:
            when = datetime.strptime(raw, "%d/%m/%Y").date()
        except ValueError:
            raise RefreshError(f"Unreadable date '{raw}' for {department}.")
        if department in current:
            blocks.append(current)
            current = {}
        current[department] = when
    if current:
        blocks.append(current)
    return blocks


def check(blocks, today: date):
    if len(blocks) != len(LEVELS):
        raise RefreshError(
            f"Expected {len(LEVELS)} lists of dates but found {len(blocks)}. "
            "The page layout may have changed."
        )
    for block in blocks:
        if not block:
            raise RefreshError("An empty list of dates was found.")
        for department, when in block.items():
            if not (today.year - 1 <= when.year <= today.year + 2):
                raise RefreshError(f"Implausible date {when} for {department}.")


def build_document(blocks, today: date) -> str:
    lines = [
        f"Karnataka State Scholarship Portal (SSP): last dates to apply for scholarships, FY {academic_year(today)}.",
        f"Collected from the SSP website on {today:%d %B %Y}. Dates can be extended, so students "
        "must confirm them on the portal.",
        "Social Welfare Department is for SC students. Tribal Welfare Department is for ST students. "
        "Backward Classes Welfare Department is for OBC, Category-1 and NT/SNT students.",
    ]
    for (_, title), block in zip(LEVELS, blocks):
        lines += ["", title]
        for department, when in block.items():
            who = f" ({WHO[department]})" if department in WHO else ""
            lines.append(f"{department}{who}: last date to submit application is {when:%d/%m/%Y}.")
    return "\n".join(lines) + "\n"


def to_json(blocks):
    return {
        key: {d: w.isoformat() for d, w in block.items()}
        for (key, _), block in zip(LEVELS, blocks)
    }


def changes(old: dict, new: dict):
    out = []
    for key, title in LEVELS:
        for department, when in new.get(key, {}).items():
            before = old.get(key, {}).get(department)
            if before is None:
                out.append(f"new: {title.split(' (')[0]} | {department}: {when}")
            elif before != when:
                out.append(f"changed: {title.split(' (')[0]} | {department}: {before} -> {when}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Refresh the SSP last dates document")
    parser.add_argument("--dry-run", action="store_true", help="show the dates, change nothing")
    parser.add_argument("--no-index", action="store_true", help="write the file but do not index it")
    args = parser.parse_args()

    today = date.today()
    try:
        blocks = parse_blocks(fetch_text())
        check(blocks, today)
    except RefreshError as exc:
        raise SystemExit(f"Not updated: {exc}")

    new = to_json(blocks)
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    old_dates = old.get("dates", {})

    print(build_document(blocks, today))
    diff = changes(old_dates, new)
    print("Changes since the last refresh:" if old_dates else "First refresh.")
    for line in diff:
        print("  " + line)
    if old_dates and not diff:
        print("  none")
    if args.dry_run:
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_document(blocks, today), encoding="utf-8")
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps({"collected": today.isoformat(), "dates": new}, indent=2), encoding="utf-8"
    )
    print(f"\nWritten: {OUTPUT}")

    if not args.no_index:
        from src.documents import update_manifest
        from src.vectorstore import index_document

        update_manifest(OUTPUT.name, {
            "scheme_name": f"SSP last dates to apply, {academic_year(today)}",
            "provider": "Government of Karnataka (SSP)",
            "category": "state",
            "academic_year": academic_year(today),
            "language": "en",
            "source_url": URL,
            "date_collected": today.isoformat(),
        })
        print("Index:", index_document(OUTPUT))


if __name__ == "__main__":
    main()