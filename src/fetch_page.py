"""Download one public page and show what text it really contains."""
import html
import re
import sys
from urllib.parse import urlparse

import httpx

from src import config

KEYWORDS = re.compile(r"last date|deadline|\b\d{1,2}/\d{1,2}/20\d\d\b|tribal welfare", re.I)


def safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def visible_text(page: str) -> str:
    page = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6]|section|table)>", "\n", page)
    page = re.sub(r"(?s)<[^>]+>", " ", page)
    page = html.unescape(page)
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in page.splitlines()]
    return "\n".join(line for line in lines if line)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.fetch_page <url>")
    url = sys.argv[1]
    try:
        response = httpx.get(
            url,
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "ScholarshipRAG-student-project/1.0"},
        )
    except Exception as exc:
        raise SystemExit(f"Download failed: {type(exc).__name__}: {exc}")

    text = visible_text(response.text)
    parts = urlparse(url)
    name = re.sub(r"[^A-Za-z0-9]+", "_", f"{parts.netloc}{parts.path}_{parts.query}").strip("_")[:80]
    out_dir = config.BASE_DIR / "data" / "processed" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.txt"
    out.write_text(text, encoding="utf-8")

    print(f"HTTP {response.status_code} | {len(response.text)} bytes of HTML | "
          f"{len(text)} characters of visible text")
    print(f"Saved to: {out}")

    hits = [line for line in text.splitlines() if KEYWORDS.search(line)]
    print(f"\n{len(hits)} line(s) mention a date, a deadline or Tribal Welfare:")
    for line in hits[:15]:
        print("  ", safe(line)[:140])

    print("\nFirst 30 lines of text:")
    for line in text.splitlines()[:30]:
        print("  ", safe(line)[:120])


if __name__ == "__main__":
    main()