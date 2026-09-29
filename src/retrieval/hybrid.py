"""
Hybrid Retrieval with Reciprocal Rank Fusion (RRF).
Fuses dense vector results and lexical BM25 rankings.
"""
from typing import List, Dict, Tuple, Any
from .vector_store import SentinelVectorStore
from .bm25_retriever import SentinelBM25Retriever
from src.ingestion.chunking import ChunkRecord

def reciprocal_rank_fusion(
    vector_results: List[Tuple[ChunkRecord, float]],
    bm25_results: List[Tuple[ChunkRecord, float]],
    k: int = 60,
    top_n: int = 10
) -> List[Tuple[ChunkRecord, float, Dict[str, Any]]]:
    """
    Combines ranked lists using Reciprocal Rank Fusion:
    RRF(d) = sum(1 / (k + rank_i(d)))
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, ChunkRecord] = {}
    rank_details: Dict[str, Dict[str, Any]] = {}

    # Process vector rankings
    for rank, (chunk, v_score) in enumerate(vector_results):
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        rrf_contrib = 1.0 / (k + rank + 1)
        scores[cid] = scores.get(cid, 0.0) + rrf_contrib
        rank_details[cid] = {
            "vector_rank": rank + 1,
            "vector_score": round(v_score, 4),
            "bm25_rank": None,
            "bm25_score": None
        }

    # Process BM25 rankings
    for rank, (chunk, b_score) in enumerate(bm25_results):
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        rrf_contrib = 1.0 / (k + rank + 1)
        scores[cid] = scores.get(cid, 0.0) + rrf_contrib
        if cid in rank_details:
            rank_details[cid]["bm25_rank"] = rank + 1
            rank_details[cid]["bm25_score"] = round(b_score, 4)
        else:
            rank_details[cid] = {
                "vector_rank": None,
                "vector_score": None,
                "bm25_rank": rank + 1,
                "bm25_score": round(b_score, 4)
            }

    # Sort descending by fused RRF score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for cid, fused_score in sorted_items[:top_n]:
        fused_results.append((
            chunk_map[cid],
            round(fused_score, 5),
            rank_details[cid]
        ))
        
    return fused_results


class SentinelHybridRetriever:
    """Unified hybrid retriever combining VectorStore and BM25 with RRF."""
    def __init__(self, vector_store: SentinelVectorStore, bm25_retriever: SentinelBM25Retriever):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def search(self, query: str, top_k: int = 10) -> List[Tuple[ChunkRecord, float, Dict[str, Any]]]:
        vector_res = self.vector_store.search(query, top_k=top_k * 2)
        bm25_res = self.bm25_retriever.search(query, top_k=top_k * 2)
        return reciprocal_rank_fusion(vector_res, bm25_res, top_n=top_k)
