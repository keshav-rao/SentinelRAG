"""
Chunking Strategies and Processing for SentinelRAG.
Provides recursive character splitting, markdown header splitting, and metadata enrichment.
"""
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loaders import SentinelDocument

logger = logging.getLogger("SentinelRAG.Chunking")


class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"


@dataclass
class ChunkRecord:
    """Represents an enriched text chunk ready for vector and BM25 indexing."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_documents(
    documents: List[SentinelDocument],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
) -> List[ChunkRecord]:
    """
    Chunks a list of SentinelDocuments into granular, metadata-enriched ChunkRecords.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )

    all_chunks: List[ChunkRecord] = []
    source_counters: Dict[str, int] = {}
    
    for doc_idx, doc in enumerate(documents):
        splits = splitter.split_text(doc.page_content)
        total_splits = len(splits)
        base_source = doc.metadata.get("source", f"doc_{doc_idx}")
        stem_name = Path(base_source).stem

        if stem_name not in source_counters:
            source_counters[stem_name] = 0

        for split_idx, split_text in enumerate(splits):
            source_counters[stem_name] += 1
            global_chunk_idx = source_counters[stem_name]
            chunk_id = f"{stem_name}_c{global_chunk_idx:03d}"
            
            # Enrich metadata
            chunk_meta = dict(doc.metadata)
            chunk_meta.update({
                "chunk_id": chunk_id,
                "chunk_index": global_chunk_idx,
                "sub_split_index": split_idx + 1,
                "char_length": len(split_text),
                "estimated_tokens": max(1, len(split_text) // 4)
            })

            all_chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                text=split_text.strip(),
                metadata=chunk_meta
            ))

    logger.info(f"Chunked {len(documents)} input documents into {len(all_chunks)} chunks.")
    return all_chunks


def save_chunks_to_json(chunks: List[ChunkRecord], output_path: Path) -> None:
    """Saves chunks as a structured JSON file for inspection or caching."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [c.to_dict() for c in chunks]
    output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks_from_json(input_path: Path) -> List[ChunkRecord]:
    """Loads previously persisted chunks from JSON."""
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")
    
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return [
        ChunkRecord(
            chunk_id=item["chunk_id"],
            text=item["text"],
            metadata=item["metadata"]
        )
        for item in data
    ]
