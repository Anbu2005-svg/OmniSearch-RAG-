import os
import time
import gc
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from retriever import FAISSMetadataRetriever
from rag_engine import RAGEngine

app = FastAPI(
    title="OmniSearch RAG Intelligence API",
    description="Vector search and RAG answer generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
engine = None
load_time = 0
_init_lock = threading.Lock()
_init_started = False


def init_engine():
    """Lazy, thread-safe initialization of retriever and engine."""
    global retriever, engine, load_time, _init_started

    if retriever is not None and engine is not None:
        return True

    if _init_started:
        return False

    with _init_lock:
        if retriever is not None and engine is not None:
            return True
        _init_started = True

    start = time.time()
    try:
        retriever = FAISSMetadataRetriever(
            index_path="faiss_index_8bit.index",
            metadata_path="faiss_metadata.jsonl",
            model_name="all-MiniLM-L6-v2"
        )
        engine = RAGEngine(retriever=retriever)
        load_time = time.time() - start
        print(f"[Server] RAG Engine initialized in {load_time:.2f}s")
        gc.collect()
        return True
    except Exception as e:
        print(f"[Server Init Error] Could not initialize RAG engine: {e}")
        with _init_lock:
            _init_started = False
        return False

@app.on_event("startup")
def startup_event():
    # Start the engine initialization in the background immediately on boot
    # This allows Uvicorn to bind to the port instantly and pass Render's health checks,
    # while the large files download in the background without hitting the 100s HTTP timeout.
    threading.Thread(target=init_engine, daemon=True).start()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0
    llm_provider: Optional[str] = ""
    api_key: Optional[str] = ""
    api_url: Optional[str] = ""
    model_name: Optional[str] = ""


class SearchResponse(BaseModel):
    answer: str
    sources: list
    query: str
    corrected_query: str
    latency: float


@app.get("/")
@app.head("/")
def root_check():
    return {"message": "OmniSearch RAG API is running live!", "docs": "/docs"}


@app.get("/api/health")
def health_check():
    ready = init_engine()
    return {
        "status": "ok" if ready else "warming_up",
        "total_vectors": retriever.total_vectors if retriever else 0,
        "vector_dim": retriever.vector_dim if retriever else 384,
        "model_name": retriever.model_name if retriever else "all-MiniLM-L6-v2",
        "load_time": round(load_time, 2)
    }


@app.get("/api/stats")
def get_stats():
    index_size_mb = 0
    if os.path.exists("faiss_index_8bit.index"):
        index_size_mb = round(os.path.getsize("faiss_index_8bit.index") / (1024 * 1024), 1)
    
    metadata_size_mb = 0
    if os.path.exists("faiss_metadata.jsonl"):
        metadata_size_mb = round(os.path.getsize("faiss_metadata.jsonl") / (1024 * 1024), 1)

    return {
        "total_vectors": retriever.total_vectors if retriever else 0,
        "vector_dim": retriever.vector_dim if retriever else 384,
        "index_size_mb": index_size_mb,
        "metadata_size_mb": metadata_size_mb,
        "model_name": retriever.model_name if retriever else "all-MiniLM-L6-v2",
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "model_configured": os.getenv("MODEL_NAME", "llama3"),
        "engine_ready": engine is not None
    }


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    ready = init_engine()
    if not ready:
        return SearchResponse(
            answer="Engine is warming up. Please wait a moment and retry.",
            sources=[],
            query=req.query,
            corrected_query=req.query,
            latency=0
        )
    
    start = time.time()
    result = engine.generate_answer(
        query=req.query,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        llm_provider=req.llm_provider,
        api_key=req.api_key,
        api_url=req.api_url,
        model_name=req.model_name
    )
    latency = time.time() - start

    return SearchResponse(
        answer=result["answer"],
        sources=result["sources"],
        query=result["query"],
        corrected_query=result.get("corrected_query", result["query"]),
        latency=round(latency, 3)
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
