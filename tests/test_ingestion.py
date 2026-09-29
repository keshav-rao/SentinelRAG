import pytest
from pathlib import Path
from src.ingestion.loaders import load_documents_from_directory, SentinelDocument
from src.ingestion.chunking import chunk_documents, ChunkingStrategy
from config.settings import settings

def test_load_sample_documents():
    docs = load_documents_from_directory(settings.raw_data_dir)
    assert len(docs) > 0, "No raw documents loaded"
    
    sources = {d.source for d in docs}
    assert "Travel_Policy.md" in sources
    assert "Leave_Policy.md" in sources
    assert "IT_Security_Policy.md" in sources

def test_chunking_preserves_metadata():
    docs = load_documents_from_directory(settings.raw_data_dir)
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= len(docs)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"

    for c in chunks:
        assert "source" in c.metadata
        assert "chunk_id" in c.metadata
        assert "estimated_tokens" in c.metadata
        assert len(c.text) > 0
