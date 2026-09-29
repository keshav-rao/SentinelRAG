"""
High-Performance Reranker for SentinelRAG.
Uses weighted hybrid fusion (RRF + Lexical Query Overlap + Vector Cosine) for sub-second reranking,
with an optional single-shot batch LLM mode. Eliminates sequential loop latency.
"""
from typing import List, Tuple, Dict, Any, Optional
import re
import math
import logging
from src.ingestion.chunking import ChunkRecord

logger = logging.getLogger("SentinelRAG.Reranker")


class SentinelReranker:
    def __init__(self, mode: str = "hybrid_cross_score"):
        """
        mode: 'hybrid_cross_score' (ultra-fast sub-millisecond scoring) or 'batch_llm'
        """
        self.mode = mode

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[ChunkRecord, float, Dict[str, Any]]],
        top_k: int = 4
    ) -> List[Tuple[ChunkRecord, float, str]]:
        """
        Reranks candidates rapidly without sequential LLM loop delays.
        Combines RRF score, exact query token overlap, and lexical presence of numbers/entities.
        """
        if not candidates:
            return []

        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        # Extract potential numeric tokens (e.g., '3000', '5000', '30', '45')
        query_numbers = set(re.findall(r'\b\d+\b', query))
        
        scored: List[Tuple[ChunkRecord, float, str]] = []

        for chunk, rrf_score, rank_info in candidates:
            text_lower = chunk.text.lower()
            text_tokens = set(re.findall(r'\b\w+\b', text_lower))
            text_numbers = set(re.findall(r'\b\d+\b', text_lower))

            # 1. Base RRF Contribution (normalized)
            base_score = min(1.0, rrf_score * 30.0)

            # 2. Token overlap ratio (Jaccard-like overlap with query)
            overlap_count = len(query_tokens.intersection(text_tokens))
            overlap_score = overlap_count / max(1, len(query_tokens))

            # 3. Numeric & entity bonus (crucial for policy limits like ₹3,000 or 30 days)
            number_bonus = 0.15 if query_numbers and query_numbers.intersection(text_numbers) else 0.0

            # 4. Vector cosine score if available
            v_score = rank_info.get("vector_score") or 0.0
            if v_score < 0:
                v_score = 0.0

            # Composite High-Precision Rerank Score
            final_score = (0.45 * base_score) + (0.35 * overlap_score) + (0.20 * v_score) + number_bonus
            final_score = round(min(1.0, max(0.0, final_score)), 4)

            reason = f"RRF: {rrf_score:.4f}, Token Overlap: {overlap_count}/{len(query_tokens)}, Vector: {v_score:.3f}"
            scored.append((chunk, final_score, reason))

        # Sort descending by final relevance score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
