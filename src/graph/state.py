"""
State definition for the SentinelRAG LangGraph workflow.
"""
from typing import TypedDict, List, Dict, Any, Optional

class SentinelGraphState(TypedDict):
    # User Input & Multi-Turn History
    question: str
    original_question: str
    thread_id: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    
    # Guardrail
    guardrail_passed: bool
    guardrail_reason: str
    
    # Retrieval
    retrieval_query: str
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    
    # Generation & Self-Evaluation
    generated_answer: str
    critic_score: float
    critic_grounded: bool
    critic_hallucination: bool
    critic_reason: str
    critic_decision: str  # "PASS" or "FAIL"
    
    # Iteration & Status
    retry_count: int
    max_retries: int
    final_status: str     # "grounded", "insufficient_evidence", "guardrail_blocked"
    sources: List[Dict[str, Any]]
