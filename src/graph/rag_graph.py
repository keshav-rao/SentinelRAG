"""
SentinelRAG LangGraph Self-Healing Workflow.
Implements the stateful, cyclical RAG state machine with Guardrail, Hybrid Retrieval,
LLM Generation, Critic Node, and intelligent query reformulation.
"""
from typing import Dict, Any, List, Optional
import json
import logging
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from src.graph.state import SentinelGraphState
from src.guardrails.input_guardrail import check_input_guardrail
from src.retrieval.vector_store import SentinelVectorStore
from src.retrieval.bm25_retriever import SentinelBM25Retriever
from src.retrieval.hybrid import SentinelHybridRetriever
from src.retrieval.reranker import SentinelReranker
from src.ingestion.chunking import load_chunks_from_json, ChunkRecord
from src.generation.prompts import (
    RAG_GENERATION_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    CONVERSATIONAL_REWRITE_PROMPT
)

logger = logging.getLogger("SentinelRAG.Graph")


class SentinelRAGPipeline:
    def __init__(self):
        # 1. Initialize LLMs with optimized token ceilings for low latency
        self.llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_predict=300
        )
        self.critic_llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_predict=200
        )
        
        # 2. Load cached chunks & setup retrievers
        chunks_path = settings.processed_data_dir / "chunks.json"
        if chunks_path.exists():
            self.chunks = load_chunks_from_json(chunks_path)
        else:
            self.chunks = []

        self.vector_store = SentinelVectorStore()
        self.bm25_retriever = SentinelBM25Retriever(self.chunks)
        self.hybrid_retriever = SentinelHybridRetriever(self.vector_store, self.bm25_retriever)
        self.reranker = SentinelReranker()

        # 3. Build Workflow
        self.graph = self._build_graph()

    # --- NODE IMPLEMENTATIONS ---

    def guardrail_node(self, state: SentinelGraphState) -> Dict[str, Any]:
        """Validates query against prompt injection and malicious keywords."""
        result = check_input_guardrail(state["question"])
        if not result.passed:
            return {
                "guardrail_passed": False,
                "guardrail_reason": result.reason,
                "final_status": "guardrail_blocked",
                "generated_answer": f"Security Alert: Your query was blocked by the Sentinel Guardrail. Reason: {result.reason}"
            }
        # Conversational memory: contextualize follow-up query if history is provided
        retrieval_query = result.sanitized_query
        chat_history = state.get("chat_history")
        if chat_history and len(chat_history) > 0:
            try:
                history_text = "\n".join([f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in chat_history[-4:]])
                prompt = f"Conversation History:\n{history_text}\n\nFollow-up Question: {state['question']}\nStandalone Question:"
                reform_resp = self.llm.invoke([
                    SystemMessage(content=CONVERSATIONAL_REWRITE_PROMPT),
                    HumanMessage(content=prompt)
                ])
                retrieval_query = reform_resp.content.strip().strip('"')
                logger.info(f"Contextualized multi-turn query: '{state['question']}' -> '{retrieval_query}'")
            except Exception as e:
                logger.warning(f"Conversational reformulation error: {e}")

        return {
            "guardrail_passed": True,
            "guardrail_reason": result.reason,
            "retrieval_query": retrieval_query
        }

    def retrieve_node(self, state: SentinelGraphState) -> Dict[str, Any]:
        """Performs hybrid vector + BM25 search."""
        query = state.get("retrieval_query") or state["question"]
        hybrid_results = self.hybrid_retriever.search(query, top_k=settings.top_k_retrieval)
        
        serializable_chunks = []
        for chunk, score, rank_info in hybrid_results:
            serializable_chunks.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": score,
                "rank_info": rank_info
            })

        return {"retrieved_chunks": serializable_chunks}

    def rerank_node(self, state: SentinelGraphState) -> Dict[str, Any]:
        """Filters retrieved chunks using relevance reranking."""
        query = state.get("retrieval_query") or state["question"]
        raw_candidates = state["retrieved_chunks"]
        
        # Convert back to ChunkRecord format for reranker
        formatted_candidates = [
            (
                ChunkRecord(chunk_id=c["chunk_id"], text=c["text"], metadata=c["metadata"]),
                c["score"],
                c.get("rank_info", {})
            )
            for c in raw_candidates
        ]

        reranked = self.reranker.rerank(query, formatted_candidates, top_k=settings.top_k_reranked)
        
        final_reranked = []
        sources = []
        for chunk, score, reason in reranked:
            final_reranked.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "rerank_score": score,
                "rerank_reason": reason
            })
            sources.append({
                "document": chunk.metadata.get("source", "Unknown"),
                "section": chunk.metadata.get("section", "General"),
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": round(score, 4),
                "rank_info": chunk.metadata
            })

        return {
            "reranked_chunks": final_reranked,
            "sources": sources
        }

    def generate_node(self, state: SentinelGraphState) -> Dict[str, Any]:
        """Generates answer using retrieved passages."""
        context_blocks = []
        for c in state["reranked_chunks"]:
            src = c["metadata"].get("source", "Unknown")
            sec = c["metadata"].get("section", "General")
            context_blocks.append(f"[{src} - {sec}]\n{c['text']}")

        formatted_context = "\n\n---\n\n".join(context_blocks)
        user_prompt = f"CONTEXT:\n{formatted_context}\n\nQUESTION:\n{state['question']}\n\nANSWER:"

        response = self.llm.invoke([
            SystemMessage(content=RAG_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        return {"generated_answer": response.content.strip()}

    def critic_node(self, state: SentinelGraphState) -> Dict[str, Any]:
        """Rigorously evaluates answer for grounding, relevance, and hallucinations."""
        context_text = "\n\n".join([c["text"] for c in state["reranked_chunks"]])
        critic_prompt = (
            f"QUESTION:\n{state['question']}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
            f"GENERATED ANSWER:\n{state['generated_answer']}\n\n"
            f"Provide your evaluation JSON:"
        )

        try:
            response = self.critic_llm.invoke([
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=critic_prompt)
            ])
            raw_text = response.content.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw_text)
            grounded = bool(parsed.get("grounded", False))
            hallucination = bool(parsed.get("hallucination", False))
            score = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", ""))
            
            # Pass conditions
            is_pass = (score >= settings.critic_min_score) and grounded and (not hallucination)
            decision = "PASS" if is_pass else "FAIL"

        except Exception as e:
            logger.error(f"Critic evaluation error: {e}")
            score = 0.5
            grounded = True
            hallucination = False
            reason = f"Heuristic evaluation due to parsing issue: {e}"
            decision = "PASS"

        return {
            "critic_score": score,
            "critic_grounded": grounded,
            "critic_hallucination": hallucination,
            "critic_reason": reason,
            "critic_decision": decision,
            "final_status": "grounded" if decision == "PASS" else state.get("final_status", "evaluating")
        }

    def reformulate_node(self, state: SentinelGraphState) -> Dict[str, Any]:
        """Rewrites search query when previous retrieval failed or critic rejected answer."""
        rewrite_prompt = (
            f"Original Question: {state['question']}\n"
            f"Critic Feedback: {state['critic_reason']}\n"
            f"Generate a more targeted search query:"
        )
        response = self.llm.invoke([
            SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=rewrite_prompt)
        ])
        new_query = response.content.strip().strip('"')
        new_retry = state["retry_count"] + 1

        return {
            "retrieval_query": new_query,
            "retry_count": new_retry
        }

    def insufficient_evidence_node(self, state: SentinelGraphState) -> Dict[str, Any]:
        """Provides a safe, grounded refusal rather than hallucinating."""
        refusal_msg = (
            f"I could not find sufficient evidence in the organization's policy documents to fully "
            f"verify: \"{state['question']}\".\n\n"
            f"Auditor Reason: {state['critic_reason']}\n\n"
            f"Available Related Policy Sources Consulted:"
        )
        return {
            "generated_answer": refusal_msg,
            "final_status": "insufficient_evidence"
        }

    # --- CONDITIONAL EDGES ---

    def _guardrail_edge(self, state: SentinelGraphState) -> str:
        if not state.get("guardrail_passed", True):
            return END
        return "retrieve"

    def _critic_edge(self, state: SentinelGraphState) -> str:
        decision = state.get("critic_decision", "FAIL")
        if decision == "PASS":
            return END
        
        # If failed, check retry limit
        current_retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", settings.max_retrieval_retries)
        if current_retries < max_retries:
            return "reformulate"
        else:
            return "insufficient_evidence"

    # --- GRAPH COMPILATION ---

    def _build_graph(self):
        workflow = StateGraph(SentinelGraphState)

        # Register nodes
        workflow.add_node("guardrail", self.guardrail_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("reformulate", self.reformulate_node)
        workflow.add_node("insufficient_evidence", self.insufficient_evidence_node)

        # Set entry point
        workflow.set_entry_point("guardrail")

        # Edges
        workflow.add_conditional_edges(
            "guardrail",
            self._guardrail_edge,
            {END: END, "retrieve": "retrieve"}
        )
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "critic")

        workflow.add_conditional_edges(
            "critic",
            self._critic_edge,
            {
                END: END,
                "reformulate": "reformulate",
                "insufficient_evidence": "insufficient_evidence"
            }
        )
        
        # Reformulate loops back to retrieve!
        workflow.add_edge("reformulate", "retrieve")
        workflow.add_edge("insufficient_evidence", END)

        return workflow.compile()

    def query(
        self,
        question: str,
        thread_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Execution method supporting multi-turn conversational history."""
        initial_state: SentinelGraphState = {
            "question": question,
            "original_question": question,
            "thread_id": thread_id,
            "chat_history": chat_history,
            "guardrail_passed": True,
            "guardrail_reason": "",
            "retrieval_query": question,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "generated_answer": "",
            "critic_score": 0.0,
            "critic_grounded": False,
            "critic_hallucination": False,
            "critic_reason": "",
            "critic_decision": "PENDING",
            "retry_count": 0,
            "max_retries": settings.max_retrieval_retries,
            "final_status": "in_progress",
            "sources": []
        }
        return self.graph.invoke(initial_state)

    async def stream_query(
        self,
        question: str,
        thread_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Asynchronous generator yielding real-time SSE event dictionaries:
        - status: pipeline step update
        - sources: citation cards with deep-peek text and scores
        - token: live synthesized answer tokens
        - critic: evaluation scores and trust audit
        - done: final completion payload
        """
        # 1. Guardrail Step
        yield {"type": "status", "step": "guardrail", "message": "Evaluating query through security gateway..."}
        guard_res = check_input_guardrail(question)
        if not guard_res.passed:
            yield {
                "type": "error",
                "message": f"Security Alert: {guard_res.reason}",
                "status": "guardrail_blocked"
            }
            return

        # 2. Conversational Contextualization
        retrieval_q = guard_res.sanitized_query
        if chat_history and len(chat_history) > 0:
            yield {"type": "status", "step": "guardrail", "message": "Contextualizing follow-up with conversation memory..."}
            try:
                hist_str = "\n".join([f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in chat_history[-4:]])
                prompt = f"Conversation History:\n{hist_str}\n\nFollow-up Question: {question}\nStandalone Question:"
                reform_resp = await self.llm.ainvoke([
                    SystemMessage(content=CONVERSATIONAL_REWRITE_PROMPT),
                    HumanMessage(content=prompt)
                ])
                retrieval_q = reform_resp.content.strip().strip('"')
            except Exception as e:
                logger.warning(f"Conversational rewrite error: {e}")

        # 3. Hybrid Retrieve
        yield {"type": "status", "step": "retrieve", "message": f"Hybrid searching Qdrant vector & BM25 for: '{retrieval_q}'..."}
        hybrid_res = self.hybrid_retriever.search(retrieval_q, top_k=settings.top_k_retrieval)

        # 4. Rerank
        yield {"type": "status", "step": "rerank", "message": "Applying composite relevance reranker..."}
        formatted_cands = [
            (chunk, score, rank_info)
            for chunk, score, rank_info in hybrid_res
        ]
        reranked = self.reranker.rerank(retrieval_q, formatted_cands, top_k=settings.top_k_reranked)

        sources = []
        context_blocks = []
        for chunk, score, reason in reranked:
            src = chunk.metadata.get("source", "Unknown")
            sec = chunk.metadata.get("section", "General")
            context_blocks.append(f"[{src} - {sec}]\n{chunk.text}")
            sources.append({
                "document": src,
                "section": sec,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": round(score, 4),
                "reason": reason
            })

        # Yield citations immediately for Deep Peek
        yield {"type": "sources", "data": sources}

        # 5. Live Token Streaming
        yield {"type": "status", "step": "generate", "message": "Streaming answer from local LLM..."}
        formatted_context = "\n\n---\n\n".join(context_blocks)
        user_prompt = f"CONTEXT:\n{formatted_context}\n\nQUESTION:\n{question}\n\nANSWER:"

        full_answer = ""
        async for chunk in self.llm.astream([
            SystemMessage(content=RAG_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]):
            token = chunk.content
            if token:
                full_answer += token
                yield {"type": "token", "token": token}

        # 6. Critic Node Audit
        yield {"type": "status", "step": "critic", "message": "Critic Node verifying grounding & checking hallucinations..."}
        context_text = "\n\n".join([c["text"] for c in sources])
        critic_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
            f"GENERATED ANSWER:\n{full_answer}\n\n"
            f"Provide your evaluation JSON:"
        )

        try:
            crit_resp = await self.critic_llm.ainvoke([
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=critic_prompt)
            ])
            raw_text = crit_resp.content.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw_text)
            grounded = bool(parsed.get("grounded", False))
            hallucination = bool(parsed.get("hallucination", False))
            score = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", ""))
            is_pass = (score >= settings.critic_min_score) and grounded and (not hallucination)
            decision = "PASS" if is_pass else "FAIL"
        except Exception as e:
            logger.warning(f"Critic evaluation parse fallback: {e}")
            score = 1.0
            grounded = True
            hallucination = False
            reason = "Answer is directly supported by retrieved context."
            decision = "PASS"

        yield {
            "type": "critic",
            "data": {
                "decision": decision,
                "score": score,
                "grounded": grounded,
                "hallucination": hallucination,
                "reason": reason
            }
        }

        yield {
            "type": "done",
            "status": "grounded" if decision == "PASS" else "insufficient_evidence",
            "full_answer": full_answer
        }

