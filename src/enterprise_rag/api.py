from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .assistant import KnowledgeAssistant
from .embeddings import MODEL_NAME, SentenceTransformerEmbedder
from .generation import ExtractiveGenerator, OllamaGenerator


REQUESTS = Counter("rag_requests_total", "RAG API requests", ["endpoint", "status"])
LATENCY = Histogram("rag_request_seconds", "RAG API latency", ["endpoint"])
RETRIEVED = Histogram("rag_retrieved_documents", "Documents supplied to generation")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    department: str | None = Field(default=None, min_length=2, max_length=80)
    top_k: int = Field(default=5, ge=1, le=10)


class WindowRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.limit = requests_per_minute
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            events = self.events[key]
            while events and events[0] < now - 60:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


_assistant: KnowledgeAssistant | None = None
_assistant_lock = threading.Lock()
rate_limiter = WindowRateLimiter(int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))


def build_assistant() -> KnowledgeAssistant:
    index_dir = Path(os.getenv("INDEX_DIR", "artifacts/vector-index-v1"))
    if not index_dir.exists():
        raise FileNotFoundError(f"index not found at {index_dir}; run `make index`")
    embedder = SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", MODEL_NAME))
    if os.getenv("GENERATOR", "extractive").casefold() == "ollama":
        generator = OllamaGenerator(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        )
    else:
        generator = ExtractiveGenerator()
    return KnowledgeAssistant.from_index(index_dir, embedder, generator)


def get_assistant() -> KnowledgeAssistant:
    global _assistant
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                _assistant = build_assistant()
    return _assistant


def authorize(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("RAG_API_KEY")
    if expected and (x_api_key is None or not secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="invalid API key")


app = FastAPI(
    title="Enterprise RAG Assistant",
    version="1.0.0",
    description="Hybrid enterprise search and grounded, citation-backed answers.",
)


@app.middleware("http")
async def observe(request: Request, call_next):
    endpoint = request.url.path
    started = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    if endpoint.startswith("/v1/") and not rate_limiter.allow(client):
        REQUESTS.labels(endpoint, "429").inc()
        return Response("rate limit exceeded", status_code=429)
    try:
        response = await call_next(request)
        REQUESTS.labels(endpoint, str(response.status_code)).inc()
        return response
    except Exception:
        REQUESTS.labels(endpoint, "500").inc()
        raise
    finally:
        LATENCY.labels(endpoint).observe(time.perf_counter() - started)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "enterprise-rag-assistant"}


@app.get("/ready")
def ready(_: None = Depends(authorize)) -> dict:
    try:
        assistant = get_assistant()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ready", "generator": assistant.generator.name, "chunks": len(assistant.index.records)}


@app.post("/v1/retrieve")
def retrieve(payload: QueryRequest, _: None = Depends(authorize)) -> dict:
    try:
        contexts = get_assistant().retrieve(payload.question, payload.top_k, payload.department)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    RETRIEVED.observe(len(contexts))
    return {"question": payload.question, "results": [context.__dict__ for context in contexts]}


@app.post("/v1/answer")
def answer(payload: QueryRequest, _: None = Depends(authorize)) -> dict:
    try:
        result = get_assistant().answer(payload.question, payload.top_k, payload.department)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    RETRIEVED.observe(len(result.citations))
    return result.to_dict()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
