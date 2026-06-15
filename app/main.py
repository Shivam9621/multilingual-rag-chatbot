"""
Phase 3: FastAPI app — exposes the RAG chain as a REST API.

Run locally:
    uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from app.rag_chain import MultilingualRAGChain


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multilingual RAG API",
    description="Cross-lingual Hindi-English RAG chatbot powered by LaBSE + Groq",
    version="1.0.0"
)

# Allow Streamlit frontend (and any origin during development) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the RAG chain once at startup (not per-request — slow to reload)
rag_chain: Optional[MultilingualRAGChain] = None


@app.on_event("startup")
def load_rag_chain():
    global rag_chain
    rag_chain = MultilingualRAGChain()


# ── Request / Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question in Hindi or English")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")


class SourceChunk(BaseModel):
    content: str
    language: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Multilingual RAG API is running",
        "docs": "/docs",
        "endpoints": ["/chat", "/health"]
    }


@app.get("/health")
def health_check():
    """Check that the RAG chain loaded successfully and ChromaDB has data."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    chunk_count = rag_chain.vectorstore._collection.count()
    return {
        "status": "healthy",
        "chunks_in_db": chunk_count,
        "embedding_model": "LaBSE",
        "llm_model": "llama-3.1-70b-versatile"
    }


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    """
    Ask a question in Hindi or English.
    Returns a grounded answer plus the source chunks used.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    try:
        result = rag_chain.query(request.question, k=request.top_k)

        sources = [
            SourceChunk(
                content=src["content"],
                language=src["metadata"].get("language", "unknown"),
                score=src["score"]
            )
            for src in result["sources"]
        ]

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
