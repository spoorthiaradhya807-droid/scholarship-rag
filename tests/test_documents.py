"""A failed upload must leave no trace: no file and no manifest row."""
import pytest

from src import config, documents


def test_failed_index_restores_manifest_and_file(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filename,scheme_name\nold.txt,Old\n", encoding="utf-8")
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(documents, "UPLOAD_DIR", raw / "uploaded")

    def boom(path):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(documents, "index_document", boom)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError):
        documents.add_document("new.txt", b"some scholarship text", {"scheme_name": "New"})

    assert manifest.read_bytes() == before
    assert not (raw / "uploaded" / "new.txt").exists()