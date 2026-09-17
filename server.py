import os
import time
import gc
import threading
import hashlib
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─── Logging (SEC-12: structured logging, no stack traces to clients) ───
logger = logging.getLogger("omnisearch")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

from retriever import FAISSMetadataRetriever
from rag_engine import RAGEngine

# ─── Environment detection ───
IS_PRODUCTION = os.getenv("RENDER") is not None or os.getenv("PORT") is not None

# ─── SEC-03: Restrict CORS to known frontend origins ───
ALLOWED_ORIGINS = [
    "https://omni-search-rag-one.vercel.app",
    "https://omni-search-rag.vercel.app",
    "http://localhost:5173",   # local Vite dev
    "http://localhost:3000",
]

# ─── SEC-09: Disable Swagger/OpenAPI docs in production ───
app = FastAPI(
    title="OmniSearch RAG Intelligence API",
    description="Vector search and RAG answer generation",
    version="1.0.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# ─── SEC-03: Fixed CORS — no wildcard + credentials combo ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ─── SEC-11 + SEC-14: Security Headers Middleware ───
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        # SEC-26: Prevent search crawlers from indexing API endpoints
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

app.add_middleware(SecurityHeadersMiddleware)


# ─── SEC-23: Request Body Size Limit Middleware (Max 1 MB) ───
MAX_REQUEST_BODY_SIZE = 1 * 1024 * 1024  # 1 MB

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BODY_SIZE:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Payload too large. Maximum request size is 1 MB."}
                    )
            except ValueError:
                pass
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware)


# ─── SEC-04: Rate Limiting (30 req/min per IP) ───
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMIT_AVAILABLE = True
    logger.info("[Security] Rate limiting enabled (slowapi)")
except ImportError:
    RATE_LIMIT_AVAILABLE = False
    logger.warning("[Security] slowapi not installed — rate limiting DISABLED")

    # Create a no-op decorator so @limiter.limit() doesn't crash
    class _NoOpLimiter:
        def limit(self, *a, **kw):
            def decorator(fn):
                return fn
            return decorator
    limiter = _NoOpLimiter()


# ─── Engine initialization ───
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
            index_path="faiss_index_8bit_50k.index",
            metadata_path="faiss_metadata_50k.jsonl",
            model_name="all-MiniLM-L6-v2"
        )
        retriever._get_encoder()
        engine = RAGEngine(retriever=retriever)
        load_time = time.time() - start
        logger.info(f"[Server] RAG Engine initialized in {load_time:.2f}s")
        gc.collect()
        return True
    except Exception as e:
        logger.error(f"[Server Init Error] Could not initialize RAG engine: {type(e).__name__}")
        with _init_lock:
            _init_started = False
        return False


@app.on_event("startup")
def startup_event():
    threading.Thread(target=init_engine, daemon=True).start()


# ─── SEC-01: SSRF Protection — allowlist of known LLM provider domains ───
ALLOWED_LLM_DOMAINS = {
    "api.openai.com",
    "api.anthropic.com",
    "api.groq.com",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.mistral.ai",
    "api.cohere.ai",
    "generativelanguage.googleapis.com",
    "ollama.com",
    "localhost",
    "127.0.0.1",
}


def validate_api_url(url: str) -> bool:
    """SEC-01: Validate that api_url points to a known LLM provider, not internal infrastructure."""
    if not url:
        return True
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        # Block private/internal IPs (AWS metadata, Docker internal, etc.)
        if hostname.startswith(("169.254.", "10.", "172.16.", "172.17.", "172.18.",
                                "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                                "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                                "172.29.", "172.30.", "172.31.", "192.168.", "0.",
                                "127.0.0.")):
            # Allow localhost only
            if hostname not in ("localhost", "127.0.0.1"):
                return False

        # Block non-HTTPS in production (except localhost)
        if IS_PRODUCTION and parsed.scheme != "https" and hostname not in ("localhost", "127.0.0.1"):
            return False

        # Check against allowlist
        if hostname not in ALLOWED_LLM_DOMAINS:
            # Allow subdomains of allowed domains
            return any(hostname.endswith(f".{domain}") for domain in ALLOWED_LLM_DOMAINS)

        return True
    except Exception:
        return False


# ─── SEC-05: Input Validation with strict Pydantic constraints ───
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_provider: Optional[str] = Field(default="", max_length=50)
    api_key: Optional[str] = Field(default="", max_length=500)
    api_url: Optional[str] = Field(default="", max_length=500)
    model_name: Optional[str] = Field(default="", max_length=100)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """SEC-06: Basic sanitization — strip control characters."""
        return "".join(c for c in v.strip() if c.isprintable() or c in ("\n", "\t"))


class SearchResponse(BaseModel):
    answer: str
    sources: list
    query: str
    corrected_query: str
    latency: float


# ─── Endpoints ───

@app.get("/")
@app.head("/")
def root_check():
    return {"message": "OmniSearch RAG API is running.", "status": "ok"}


@app.get("/ping")
@app.head("/ping")
def ping():
    """Minimal liveness probe — no engine init required."""
    return {"status": "pong"}


# ─── SEC-10: Minimal health check — don't leak internal details ───
@app.get("/health")
@app.get("/healthz")
@app.get("/actuator/health")
@app.get("/api/health")
def health_check():
    ready = init_engine()
    return {
        "status": "ok" if ready else "warming_up",
        "ready": ready,
    }


@app.get("/api/stats")
def get_stats():
    """SEC-10: Reduced information exposure — no model names or file sizes."""
    return {
        "total_vectors": retriever.total_vectors if retriever else 0,
        "vector_dim": retriever.vector_dim if retriever else 384,
        "engine_ready": engine is not None,
    }


@app.post("/api/search", response_model=SearchResponse)
@limiter.limit("30/minute")
def search(req: SearchRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    client_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:10]

    # SEC-01: Validate api_url before making any outbound requests
    if req.api_url and not validate_api_url(req.api_url):
        logger.warning(f"[Audit] Client={client_hash} BlockedURL={req.api_url[:50]} Status=403")
        return SearchResponse(
            answer="⚠️ Security Error: The provided API URL is not allowed. Only known LLM provider endpoints are permitted.",
            sources=[],
            query=req.query,
            corrected_query=req.query,
            latency=0,
        )

    ready = init_engine()
    if not ready:
        logger.info(f"[Audit] Client={client_hash} WarmingUp=True Status=503")
        return SearchResponse(
            answer="Engine is warming up. Please wait a moment and retry.",
            sources=[],
            query=req.query,
            corrected_query=req.query,
            latency=0,
        )

    start = time.time()
    try:
        result = engine.generate_answer(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
            llm_provider=req.llm_provider,
            api_key=req.api_key,
            api_url=req.api_url,
            model_name=req.model_name,
        )
        latency = time.time() - start

        # SEC-24: Structured audit log with hashed client IP and latency
        logger.info(f"[Audit] Client={client_hash} QueryLen={len(req.query)} TopK={req.top_k} Latency={latency:.3f}s Status=200")

        return SearchResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=result["query"],
            corrected_query=result.get("corrected_query", result["query"]),
            latency=round(latency, 3),
        )
    except Exception:
        logger.exception(f"[Audit] Client={client_hash} Status=500 Unexpected search error")
        return SearchResponse(
            answer="An internal error occurred. Please try again.",
            sources=[],
            query=req.query,
            corrected_query=req.query,
            latency=0,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
