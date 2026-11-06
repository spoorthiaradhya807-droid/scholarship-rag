"""Index documents into ChromaDB, list them, or test a search."""
import argparse

from src.embeddings import EmbeddingError, embed_query
from src.vectorstore import get_collection, list_documents, search, sync_all


def safe(text: str) -> str:
    """The Windows console cannot show Kannada, so show it as '?'."""
    return text.encode("ascii", "replace").decode("ascii")


def show_search(query: str, k: int):
    hits = search(embed_query(query), k)
    if not hits:
        print("No results. Is the database empty?")
        return
    for rank, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        snippet = " ".join(hit["text"].split())[:170]
        print(
            f"{rank}. similarity {hit['similarity']:.3f} | {meta['source']} | "
            f"page {meta['page']} | {meta['reliability']}\n   {safe(snippet)}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Manage the scholarship vector database")
    parser.add_argument("--only", help="only files whose name contains this text")
    parser.add_argument("--dry-run", action="store_true", help="show what would change, no API calls")
    parser.add_argument("--list", action="store_true", help="list indexed documents")
    parser.add_argument("--query", help="test a search with this question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        if args.query:
            show_search(args.query, args.top_k)
        elif args.list:
            for d in list_documents():
                print(f"{d['source']} | chunks={d['chunks']} pages={d['pages']} | {d['reliability']}")
            print(f"\n{len(list_documents())} documents, {get_collection().count()} chunks")
        else:
            summary, errors = sync_all(only=args.only, dry_run=args.dry_run)
            print(f"\nSummary: {summary or 'nothing to do'}")
            if errors:
                print(f"{len(errors)} file(s) had errors (see above).")
            print(f"Chunks in database: {get_collection().count()}")
    except EmbeddingError as exc:
        print(f"\nStopped: {exc}")
        print("Documents already finished are saved. Run the same command again to continue.")


if __name__ == "__main__":
    main()