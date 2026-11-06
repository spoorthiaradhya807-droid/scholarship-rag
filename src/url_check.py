"""Compare two ways of reading an official page:
1. Gemini's URL context tool (Google fetches the page)
2. a direct download from this computer (httpx)
"""
import re
import sys

import httpx
from google import genai
from google.genai import types

from src import config

DEFAULT_URLS = [
    "https://www.aicte-india.org/schemes/students-development-schemes",
    "https://www.education.gov.in/node/255",
    "https://scholarships.gov.in/",
    "https://ssp.karnataka.gov.in/?section=schemes",
]
QUESTION = (
    "List the scholarship scheme names that are on this page. Use only the page content. "
    "If you cannot read the page, say exactly: PAGE NOT READABLE."
)


def safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def local_fetch(url: str) -> str:
    try:
        response = httpx.get(
            url,
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "ScholarshipRAG-student-project/1.0"},
        )
    except Exception as exc:
        return f"download failed: {type(exc).__name__}"
    html = response.text
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (
        f"HTTP {response.status_code}, {len(html)} bytes of HTML, "
        f"{len(text)} characters of visible text"
    )


def gemini_fetch(client, url: str):
    try:
        interaction = client.interactions.create(
            model=config.GEN_MODEL,
            input=f"{QUESTION} Page: {url}",
            tools=[{"type": "url_context"}],
        )
    except Exception as exc:
        return [f"error: {type(exc).__name__}: {exc}"], ""
    statuses, answer = [], ""
    for step in interaction.steps:
        if step.type == "url_context_result":
            try:
                statuses.append(step.model_dump_json(exclude_none=True)[:300])
            except Exception:
                statuses.append(repr(step)[:300])
        if step.type == "model_output":
            for block in step.content:
                if block.type == "text":
                    answer += block.text
    return statuses, answer


def main():
    urls = sys.argv[1:] or DEFAULT_URLS
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=120_000),
    )
    for url in urls:
        print("=" * 70)
        print(url)
        print("  direct download:", local_fetch(url))
        statuses, answer = gemini_fetch(client, url)
        for status in list(dict.fromkeys(statuses))[:2]:
            print("  gemini fetch result:", safe(status))
        print("  gemini answer:", safe(" ".join(answer.split()))[:260])


if __name__ == "__main__":
    main()