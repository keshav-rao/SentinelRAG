"""
Pydantic Schemas for SentinelRAG FastAPI service.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class MessageHistoryItem(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class QueryRequest(BaseModel):
    question: str = Field(..., description="The user query to evaluate against the knowledge base")
    user_id: Optional[str] = Field(default="EMP1024", description="Employee or session identifier")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread identifier for multi-turn history")
    chat_history: Optional[List[MessageHistoryItem]] = Field(default=None, description="Recent conversation turns")

class SourceCitation(BaseModel):
    document: str
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    text: Optional[str] = Field(default=None, description="Full passage text for deep inspection")
    score: Optional[float] = Field(default=None, description="Relevance score (0.0 to 1.0)")
    rank_info: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    status: str = Field(..., description="'grounded', 'insufficient_evidence', or 'guardrail_blocked'")
    confidence: float
    sources: List[SourceCitation]
    retrieval_attempts: int
    critic_decision: str
    critic_reason: Optional[str] = None
    thread_id: Optional[str] = None

class UploadResponse(BaseModel):
    status: str
    filename: str
    file_size_bytes: int
    chunks_created: int
    indexed_in_qdrant: int
    message: str

class IngestionResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    indexed_in_qdrant: int
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_model: str
    qdrant_collection: str
    langsmith_tracing: bool
