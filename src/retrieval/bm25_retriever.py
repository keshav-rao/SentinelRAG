"""
BM25 Lexical Retriever for SentinelRAG using rank_bm25.
Complements dense vector search by catching exact acronyms, numbers (e.g., ₹3,000, 30 days), and policy codes.
"""
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi
from src.ingestion.chunking import ChunkRecord

class SentinelBM25Retriever:
    def __init__(self, chunks: List[ChunkRecord]):
        self.chunks = chunks
        self.corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric (preserves numeric values and codes)
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def search(self, query: str, top_k: int = 10) -> List[Tuple[ChunkRecord, float]]:
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair with chunks and sort
        scored_pairs = list(zip(self.chunks, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [(chunk, float(score)) for chunk, score in scored_pairs[:top_k] if score > 0]
