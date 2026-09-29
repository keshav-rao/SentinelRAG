"""
Qdrant Vector Store indexing and similarity search using Ollama embeddings.
Operates in local embedded storage mode (no external Docker dependency required).
"""
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_ollama import OllamaEmbeddings

from config.settings import settings
from src.ingestion.chunking import ChunkRecord

logger = logging.getLogger("SentinelRAG.VectorStore")


class SentinelVectorStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        local_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.local_path = local_path or settings.qdrant_local_path
        
        # Ensure local storage directory exists
        Path(self.local_path).mkdir(parents=True, exist_ok=True)
        
        self.embeddings = OllamaEmbeddings(
            model=embedding_model or settings.ollama_embed_model,
            base_url=base_url or settings.ollama_base_url
        )

        try:
            self.client = QdrantClient(path=self.local_path)
            self._is_fallback = False
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Local Qdrant storage is locked by active server: {e}. Initializing ephemeral mirror.")
            self.client = QdrantClient(":memory:")
            self._is_fallback = True
            self._ensure_collection()
            # Fast populate from chunks.json if available
            self._populate_mirror()

    def _populate_mirror(self):
        chunks_path = settings.processed_data_dir / "chunks.json"
        if chunks_path.exists():
            from src.ingestion.chunking import load_chunks_from_json
            cached = load_chunks_from_json(chunks_path)
            self.index_chunks(cached)

    def _ensure_collection(self):
        # We determine embedding dimension dynamically or default to 768 for nomic-embed-text
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            # Query dimension sample
            sample_vec = self.embeddings.embed_query("dimension probe")
            dim = len(sample_vec)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' with vector size {dim}")

    def index_chunks(self, chunks: List[ChunkRecord], batch_size: int = 16) -> int:
        """Embeds and indexes chunk records into Qdrant."""
        if not chunks:
            return 0
        
        points = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            vectors = self.embeddings.embed_documents(texts)
            
            for chunk_idx, (chunk, vec) in enumerate(zip(batch, vectors)):
                point_id = i + chunk_idx + 1
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata
                }
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload
                ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} chunks into collection '{self.collection_name}'")
        return len(points)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[ChunkRecord, float]]:
        """Dense semantic search returning (ChunkRecord, similarity_score)."""
        query_vec = self.embeddings.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k
        )
        
        scored_chunks = []
        for res in response.points:
            payload = res.payload or {}
            chunk = ChunkRecord(
                chunk_id=payload.get("chunk_id", ""),
                text=payload.get("text", ""),
                metadata=payload.get("metadata", {})
            )
            scored_chunks.append((chunk, float(res.score)))
            
        return scored_chunks
