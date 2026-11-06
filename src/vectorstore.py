"""ChromaDB storage: add, replace, remove and search document chunks."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import chromadb

from src import config
from src.chunking import chunk_document
from src.embeddings import embed_documents
from src.loaders import SUPPORTED_EXTENSIONS, DocumentLoadError, load_manifest

ADD_BATCH = 100
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )
    return _collection


def fingerprint(chunks) -> str:
    """One hash for a document's chunks, text, metadata and embedding settings."""
    digest = hashlib.sha256()
    for c in chunks:
        digest.update(c.chunk_id.encode("utf-8"))
        digest.update(c.embed_text.encode("utf-8"))
        digest.update(json.dumps(c.to_metadata(), sort_keys=True).encode("utf-8"))
    digest.update(f"{config.EMBED_MODEL}:{config.EMBED_DIMENSIONS}".encode("utf-8"))
    return digest.hexdigest()


def index_document(path, manifest=None, dry_run: bool = False) -> str:
    """Add or update one document. Returns unchanged / new / changed (dry run)
    or indexed / updated."""
    path = Path(path)
    chunks = chunk_document(path, manifest)
    if not chunks:
        raise DocumentLoadError(f"{path.name}: no usable text after cleaning.")

    new_hash = fingerprint(chunks)
    col = get_collection()
    existing = col.get(where={"source": path.name}, include=["metadatas"])
    old_ids = existing["ids"]
    old_hashes = {m.get("doc_hash") for m in existing["metadatas"]}

    if len(old_ids) == len(chunks) and old_hashes == {new_hash}:
        return "unchanged"
    if dry_run:
        return "changed" if old_ids else "new"

    # Embed first, so a failed API call never leaves the document half-deleted.
    vectors = embed_documents([c.embed_text for c in chunks])
    if old_ids:
        col.delete(ids=old_ids)
    for start in range(0, len(chunks), ADD_BATCH):
        part = chunks[start:start + ADD_BATCH]
        col.upsert(
            ids=[c.chunk_id for c in part],
            embeddings=vectors[start:start + ADD_BATCH],
            documents=[c.text for c in part],
            metadatas=[{**c.to_metadata(), "doc_hash": new_hash} for c in part],
        )
    return "updated" if old_ids else "indexed"


def remove_document(source_name: str) -> int:
    """Delete every chunk of one document. Returns how many were removed."""
    col = get_collection()
    ids = col.get(where={"source": source_name})["ids"]
    if ids:
        col.delete(ids=ids)
    return len(ids)


def list_documents():
    """One entry per indexed document, for the admin screen."""
    data = get_collection().get(include=["metadatas"])
    docs = defaultdict(lambda: {"chunks": 0, "pages": set(), "meta": {}})
    for meta in data["metadatas"]:
        entry = docs[meta["source"]]
        entry["chunks"] += 1
        entry["pages"].add(meta.get("page", 0))
        entry["meta"] = meta
    return [
        {
            "source": name,
            "chunks": entry["chunks"],
            "pages": len(entry["pages"]),
            "scheme_name": entry["meta"].get("scheme_name", ""),
            "academic_year": entry["meta"].get("academic_year", ""),
            "reliability": entry["meta"].get("reliability", ""),
        }
        for name, entry in sorted(docs.items())
    ]


def search(query_vector, k: int = 6, where=None):
    """Return the k most similar chunks. Cosine distance: lower means closer."""
    col = get_collection()
    total = col.count()
    if total == 0:
        return []
    result = col.query(
        query_embeddings=[query_vector],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    hits = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {"text": text, "metadata": meta, "distance": dist, "similarity": 1 - dist}
        )
    return hits


def sync_all(only=None, dry_run: bool = False, remove_missing: bool = True):
    """Make the database match data/raw: add new files, update changed ones,
    and remove documents whose file was deleted."""
    manifest = load_manifest()
    files = sorted(
        p for p in config.RAW_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    on_disk = {p.name for p in files}
    if only:
        files = [p for p in files if only.lower() in p.name.lower()]

    summary, errors = defaultdict(int), []
    for path in files:
        try:
            status = index_document(path, manifest, dry_run)
        except DocumentLoadError as exc:
            errors.append(str(exc))
            print(f"  ERROR {exc}")
            continue
        summary[status] += 1
        if status != "unchanged":
            print(f"  {status}: {path.name}")

    if remove_missing and not only:
        for name in sorted({d["source"] for d in list_documents()} - on_disk):
            if not dry_run:
                remove_document(name)
            print(f"  {'would remove' if dry_run else 'removed'}: {name}")
            summary["removed"] += 1
    return dict(summary), errors