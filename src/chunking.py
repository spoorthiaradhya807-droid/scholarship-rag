"""Clean page text and split it into chunks that keep source and page number."""
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path



from src import config
from src.loaders import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    load_document,
    load_manifest,
)

CHUNK_SIZE = 1000            # characters per chunk (about 150-250 words)
CHUNK_OVERLAP = 150          # shared characters between neighbouring chunks
MIN_CHUNK_CHARS = 40         # smaller fragments are dropped
MIN_PAGES_FOR_EDGE_CHECK = 6
EDGE_REPEAT_SHARE = 0.6      # a header/footer line must be on 60% of pages

NAV_WORDS = ("skip to content", "screen reader", "downloads", "rti", "search...", "search")
SESSION_IP = re.compile(r"session ip\s*:?\s*\S*", re.I)
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
TIMESTAMP_START = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\s*", re.I
)
URL_PAGE_LINE = re.compile(r"^https?://\S+\s+\d+/\d+$", re.I)
PAGE_NUMBER_LINE = re.compile(r"^(?:page\s*)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?$", re.I)

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_recursive(text, separators, size):
    """Cut text into pieces of at most `size` characters, splitting at the
    largest separator that works (paragraph, line, sentence, word, character)."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep == "":
            return [text[j:j + size] for j in range(0, len(text), size)]
        if sep in text:
            parts = text.split(sep)
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]   # keep the separator
            pieces = []
            for part in parts:
                if len(part) <= size:
                    pieces.append(part)
                else:
                    pieces.extend(_split_recursive(part, separators[i + 1:], size))
            return pieces
    return [text]


def _merge(pieces, size, overlap):
    """Join small pieces into chunks of up to `size`, repeating a short tail
    (up to `overlap` characters) at the start of the next chunk."""
    chunks, current, length = [], [], 0
    for piece in pieces:
        if current and length + len(piece) > size:
            chunks.append("".join(current))
            while current and (length > overlap or length + len(piece) > size):
                length -= len(current[0])
                current.pop(0)
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current))
    return chunks


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one page of text into overlapping chunks."""
    return _merge(_split_recursive(text, SEPARATORS, size), size, overlap)

@dataclass
class Chunk:
    chunk_id: str        # stable ID: file::page::number
    source: str          # file name, shown in citations
    page: int            # 1-based page number, 0 if the file has no pages
    text: str
    text_source: str     # native / ocr / manual / docx / txt
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: a short context line plus the chunk."""
        parts = (
            self.metadata.get("scheme_name"),
            self.metadata.get("provider"),
            self.metadata.get("academic_year"),
        )
        head = " | ".join(p for p in parts if p)
        return f"{head}\n{self.text}" if head else self.text

    def to_metadata(self) -> dict:
        """Flat values only (the vector database does not accept None or lists)."""
        meta = {
            "source": self.source,
            "page": self.page,
            "text_source": self.text_source,
            "reliability": reliability_label(self.source),
        }
        for key, value in self.metadata.items():
            meta[key] = value or ""
        return meta


def reliability_label(source: str) -> str:
    """Screenshots only show whatever was on screen, so mark them lower."""
    if "screenshot" in source or source.startswith("ssp_dept_"):
        return "screenshot"
    return "official"


def normalize_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.lower())).strip()


def is_nav_line(line: str) -> bool:
    """True if the line holds only website menu words and symbols."""
    rest = line.lower()
    for word in NAV_WORDS:
        rest = rest.replace(word, " ")
    return re.sub(r"[\W_]+", "", rest) == ""


def is_boilerplate_line(line: str) -> bool:
    return bool(
        URL_PAGE_LINE.match(line) or PAGE_NUMBER_LINE.match(line) or is_nav_line(line)
    )


def clean_page_text(text: str):
    """Return (cleaned text, number of removed lines)."""
    kept, removed = [], 0
    for raw in text.splitlines():
        line = SESSION_IP.sub("", raw)
        line = TIMESTAMP_START.sub("", line)
        line = IPV4.sub("", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not raw.strip():
            kept.append("")
        elif not line or is_boilerplate_line(line):
            removed += 1
        else:
            kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), removed


def strip_repeated_edges(texts):
    """Remove header/footer lines repeated on most pages of a long document."""
    if len(texts) < MIN_PAGES_FOR_EDGE_CHECK:
        return texts, 0

    split = [t.split("\n") for t in texts]
    edge_sets, counts = [], Counter()
    for lines in split:
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        if len(idx) < 8:                      # short pages: leave alone
            edge_sets.append(set())
            continue
        edges = set(idx[:2] + idx[-2:])
        edge_sets.append(edges)
        counts.update({normalize_line(lines[i]) for i in edges})

    limit = max(3, math.ceil(EDGE_REPEAT_SHARE * len(texts)))
    repeated = {n for n, c in counts.items() if c >= limit and 0 < len(n) <= 100}

    removed, result = 0, []
    for lines, edges in zip(split, edge_sets):
        out = []
        for i, ln in enumerate(lines):
            if i in edges and normalize_line(ln) in repeated:
                removed += 1
                continue
            out.append(ln)
        result.append("\n".join(out))
    return result, removed


def chunk_document(path, manifest: dict = None, stats: dict = None) -> list:
    """Load, clean and split one document. Raises DocumentLoadError if unreadable."""
    path = Path(path)
    pages = load_document(path, manifest, include_empty=False)

    removed_total, cleaned = 0, []
    for page in pages:
        text, removed = clean_page_text(page.text)
        removed_total += removed
        cleaned.append(text)
    cleaned, removed_edges = strip_repeated_edges(cleaned)
    removed_total += removed_edges

    chunks = []
    for page, text in zip(pages, cleaned):
        text = text.strip()
        if not text:
            continue
        number = page.page or 0
        count = 0
        for piece in split_text(text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            count += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{page.source}::p{number}::c{count}",
                    source=page.source,
                    page=number,
                    text=piece,
                    text_source=page.text_source,
                    metadata=dict(page.metadata),
                )
            )
    if stats is not None:
        stats["removed_lines"] = removed_total
    return chunks


def main():
    manifest = load_manifest()
    files = sorted(
        p for p in config.RAW_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    lines, preview = [], []
    all_chunks = []
    for path in files:
        stats = {}
        try:
            chunks = chunk_document(path, manifest, stats)
        except DocumentLoadError as exc:
            lines.append(f"{path.name}: ERROR {exc}")
            continue
        all_chunks.extend(chunks)
        avg = round(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0
        lines.append(
            f"{path.name} | chunks={len(chunks)} avg_chars={avg} "
            f"removed_lines={stats.get('removed_lines', 0)}"
        )
        for c in chunks[:2]:
            preview.append(f"=== {c.source} | page {c.page} | {c.text_source} ===\n{c.text}\n")

    ids = [c.chunk_id for c in all_chunks]
    longest = max((len(c.text) for c in all_chunks), default=0)
    lines.append("")
    lines.append(
        f"TOTAL: {len(files)} files, {len(all_chunks)} chunks, "
        f"longest chunk {longest} chars, duplicate IDs: {len(ids) - len(set(ids))}"
    )

    out_dir = config.BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunk_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "chunks_preview.txt").write_text("\n".join(preview), encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode("ascii"))
    print("\nPreview of the first 2 chunks per file: data\\processed\\chunks_preview.txt")


if __name__ == "__main__":
    main()