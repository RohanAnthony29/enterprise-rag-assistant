from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .embeddings import Embedder
from .generation import Context, Generator
from .hybrid import HybridRetriever
from .lexical import tokenize
from .vector_index import NumpyVectorIndex


@dataclass(frozen=True)
class AssistantResponse:
    question: str
    answer: str
    citations: list[dict]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    generator: str

    def to_dict(self) -> dict:
        return asdict(self)


class KnowledgeAssistant:
    def __init__(self, index: NumpyVectorIndex, embedder: Embedder, generator: Generator):
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.retriever = HybridRetriever(index)

    @classmethod
    def from_index(cls, index_dir: Path, embedder: Embedder, generator: Generator):
        return cls(NumpyVectorIndex.load(index_dir), embedder, generator)

    def retrieve(self, question: str, top_k: int = 5, department: str | None = None) -> list[Context]:
        query_vector = self.embedder.encode([question])[0]
        document_hits = self.retriever.search(
            question, query_vector, top_k=top_k, department=department
        )
        query_terms = set(tokenize(question))
        contexts: list[Context] = []
        for rank, hit in enumerate(document_hits, start=1):
            best_index, best_score = None, float("-inf")
            for index, record in enumerate(self.index.records):
                if record["metadata"]["repository_path"] != hit.repository_path:
                    continue
                terms = set(tokenize(record["text"]))
                lexical_overlap = len(query_terms & terms) / max(1, len(query_terms))
                dense_score = float(self.index.embeddings[index] @ query_vector)
                score = 0.8 * dense_score + 0.2 * lexical_overlap
                if score > best_score:
                    best_index, best_score = index, score
            if best_index is None:
                continue
            record = self.index.records[best_index]
            contexts.append(
                Context(
                    rank=rank,
                    document_id=hit.document_id,
                    title=hit.title,
                    text=record["text"],
                    repository_path=hit.repository_path,
                    source_url=hit.source_url,
                    department=hit.department,
                    score=hit.score,
                )
            )
        return contexts

    def answer(self, question: str, top_k: int = 5, department: str | None = None) -> AssistantResponse:
        started = time.perf_counter()
        contexts = self.retrieve(question, top_k, department)
        retrieved = time.perf_counter()
        answer = self.generator.generate(question, contexts)
        finished = time.perf_counter()
        return AssistantResponse(
            question=question,
            answer=answer,
            citations=[
                {
                    "id": context.rank,
                    "document_id": context.document_id,
                    "title": context.title,
                    "department": context.department,
                    "repository_path": context.repository_path,
                    "source_url": context.source_url,
                    "score": round(context.score, 6),
                }
                for context in contexts
            ],
            retrieval_ms=round((retrieved - started) * 1000, 3),
            generation_ms=round((finished - retrieved) * 1000, 3),
            total_ms=round((finished - started) * 1000, 3),
            generator=self.generator.name,
        )
