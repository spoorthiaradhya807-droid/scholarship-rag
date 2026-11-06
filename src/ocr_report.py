"""Summarize OCR output so we can check quality and fix file names."""
import json

from src import config


def main():
    files = sorted(config.OCR_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No OCR files found in data/processed/ocr/")

    lines = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        pages = data["pages"]
        texts = [pages[k] for k in sorted(pages, key=int)]
        total_chars = sum(len(t) for t in texts)
        short = sorted((k for k, t in pages.items() if len(t.strip()) < 40), key=int)
        first = " ".join(texts[0].split())[:200] if texts else ""

        flags = []
        if short:
            flags.append("short/empty pages: " + ", ".join(short))
        if any("session ip" in t.lower() for t in texts):
            flags.append("contains 'Session IP'")

        lines.append(
            f"{data['source']}\n"
            f"  pages OCR'd: {len(pages)} | characters: {total_chars} | "
            f"model: {data.get('ocr_model')}\n"
            f"  starts with: {first}\n"
            f"  flags: {'; '.join(flags) or 'none'}\n"
        )

    report = "\n".join(lines)
    out = config.BASE_DIR / "data" / "processed" / "ocr_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"{len(files)} OCR files. Full report saved to {out}\n")
    # The Windows console cannot show Kannada, so it prints as '?' here.
    print(report.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()