"""
SentinelRAG FastAPI Application.
Exposes REST and SSE streaming endpoints for querying, file uploading, and health checks.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from config.settings import settings
from api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestionResponse,
    HealthResponse,
    SourceCitation,
    UploadResponse
)
from src.graph.rag_graph import SentinelRAGPipeline
from src.guardrails.output_guardrail import check_output_guardrail
from src.ingestion.loaders import load_document, load_documents_from_directory
from src.ingestion.chunking import chunk_documents, save_chunks_to_json, load_chunks_from_json, ChunkingStrategy
from src.retrieval.vector_store import SentinelVectorStore

logger = logging.getLogger("SentinelRAG.API")

app = FastAPI(
    title="SentinelRAG — Self-Evaluating Enterprise RAG",
    description="Hallucination-resistant Enterprise RAG with LangGraph, hybrid retrieval, SSE streaming, and Critic node self-healing.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global pipeline instance
pipeline: Optional[SentinelRAGPipeline] = None

@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info("Starting SentinelRAG service and initializing graph...")
    pipeline = SentinelRAGPipeline()

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        ollama_model=settings.ollama_llm_model,
        qdrant_collection=settings.qdrant_collection_name,
        langsmith_tracing=settings.langsmith_tracing
    )

@app.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    global pipeline
    if not pipeline:
        pipeline = SentinelRAGPipeline()

    try:
        history_dicts = [m.model_dump() for m in request.chat_history] if request.chat_history else None
        result = pipeline.query(
            question=request.question,
            thread_id=request.thread_id,
            chat_history=history_dicts
        )
        
        # Output Guardrail validation
        raw_answer = result.get("generated_answer", "")
        raw_sources = result.get("sources", [])
        guard_res = check_output_guardrail(raw_answer, raw_sources)

        citations = [
            SourceCitation(
                document=s.get("document", "Unknown"),
                section=s.get("section"),
                chunk_id=s.get("chunk_id"),
                text=s.get("text"),
                score=s.get("score"),
                rank_info=s.get("rank_info")
            )
            for s in raw_sources
        ]

        return QueryResponse(
            question=request.question,
            answer=guard_res.sanitized_answer,
            status=result.get("final_status", "grounded"),
            confidence=round(result.get("critic_score", 0.0), 2),
            sources=citations,
            retrieval_attempts=result.get("retry_count", 0) + 1,
            critic_decision=result.get("critic_decision", "PASS"),
            critic_reason=result.get("critic_reason"),
            thread_id=request.thread_id
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG pipeline: {str(e)}"
        )

@app.post("/rag/query/stream")
async def stream_query_rag(request: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint for instantaneous token response."""
    global pipeline
    if not pipeline:
        pipeline = SentinelRAGPipeline()

    history_dicts = [m.model_dump() for m in request.chat_history] if request.chat_history else None

    async def event_generator():
        try:
            async for event in pipeline.stream_query(
                question=request.question,
                thread_id=request.thread_id,
                chat_history=history_dicts
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/rag/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, DOCX, MD, or TXT policy document, chunks it, and indexes it into Qdrant."""
    global pipeline
    allowed_extensions = {".pdf", ".docx", ".md", ".txt", ".markdown"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(allowed_extensions)}"
        )

    # Save file to data/raw
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    saved_path = settings.raw_data_dir / file.filename
    content = await file.read()
    saved_path.write_bytes(content)

    try:
        # 1. Parse document
        docs = load_document(saved_path)
        if not docs:
            raise ValueError(f"Could not extract any content from {file.filename}")

        # 2. Chunk
        new_chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=80)
        
        # 3. Append to existing chunks cache
        chunks_path = settings.processed_data_dir / "chunks.json"
        existing_chunks = []
        if chunks_path.exists():
            existing_chunks = load_chunks_from_json(chunks_path)
            
        # Filter out existing chunks from the same file if re-uploaded
        filtered_existing = [c for c in existing_chunks if c.metadata.get("source") != file.filename]
        all_chunks = filtered_existing + new_chunks
        save_chunks_to_json(all_chunks, chunks_path)

        # 4. Index into Qdrant
        vstore = SentinelVectorStore()
        indexed_count = vstore.index_chunks(new_chunks)

        # 5. Refresh pipeline instance
        pipeline = SentinelRAGPipeline()

        return UploadResponse(
            status="success",
            filename=file.filename,
            file_size_bytes=len(content),
            chunks_created=len(new_chunks),
            indexed_in_qdrant=indexed_count,
            message=f"Successfully processed and indexed '{file.filename}' with {len(new_chunks)} chunks."
        )
    except Exception as e:
        logger.error(f"Error processing uploaded file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {str(e)}"
        )

@app.post("/rag/ingest", response_model=IngestionResponse)
def trigger_ingestion():
    try:
        raw_docs = load_documents_from_directory(settings.raw_data_dir)
        if not raw_docs:
            return IngestionResponse(
                status="warning",
                documents_loaded=0,
                chunks_created=0,
                indexed_in_qdrant=0,
                message="No raw documents found in data/raw/"
            )

        chunks = chunk_documents(raw_docs, chunk_size=500, chunk_overlap=80)
        chunks_path = settings.processed_data_dir / "chunks.json"
        save_chunks_to_json(chunks, chunks_path)

        vstore = SentinelVectorStore()
        count = vstore.index_chunks(chunks)

        global pipeline
        pipeline = SentinelRAGPipeline()

        return IngestionResponse(
            status="success",
            documents_loaded=len(raw_docs),
            chunks_created=len(chunks),
            indexed_in_qdrant=count,
            message=f"Successfully indexed {count} vectors into Qdrant collection '{settings.qdrant_collection_name}'."
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

# Mount Frontend static files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
