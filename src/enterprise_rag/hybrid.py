from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lexical import BM25Index, tokenize
from .vector_index import NumpyVectorIndex


@dataclass(frozen=True)
class HybridHit:
    document_id: str
    repository_path: str
    title: str
    department: str
    source_url: str
    score: float
    dense_score: float
    lexical_score: float
    rrf_score: float
    title_overlap: float


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


class HybridRetriever:
    def __init__(
        self,
        vector_index: NumpyVectorIndex,
        dense_weight: float = 0.50,
        lexical_weight: float = 0.30,
        rrf_weight: float = 0.10,
        title_weight: float = 0.10,
    ):
        self.vector_index = vector_index
        self.records = vector_index.records
        self.lexical_index = BM25Index(self.records)
        self.weights = (dense_weight, lexical_weight, rrf_weight, title_weight)

    @staticmethod
    def _collapse_by_document(items, path_for, score_for) -> tuple[list[str], dict[str, float]]:
        ranking, scores, seen = [], {}, set()
        for item in items:
            path = path_for(item)
            scores[path] = max(scores.get(path, float("-inf")), score_for(item))
            if path not in seen:
                seen.add(path)
                ranking.append(path)
        return ranking, scores

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        department: str | None = None,
        candidate_k: int = 150,
    ) -> list[HybridHit]:
        dense_hits = self.vector_index.search(query_vector, candidate_k, department)
        lexical_hits = self.lexical_index.search(query, candidate_k, department)
        dense_ranking, dense_scores = self._collapse_by_document(
            dense_hits,
            lambda hit: hit.metadata["repository_path"],
            lambda hit: hit.score,
        )
        lexical_ranking, lexical_scores = self._collapse_by_document(
            lexical_hits,
            lambda hit: self.records[hit.record_index]["metadata"]["repository_path"],
            lambda hit: hit.score,
        )
        rrf: dict[str, float] = {}
        for ranking in (dense_ranking, lexical_ranking):
            for rank, path in enumerate(ranking, start=1):
                rrf[path] = rrf.get(path, 0.0) + 1.0 / (60 + rank)
        candidates = set(dense_ranking) | set(lexical_ranking)
        dense_normalized = _minmax({path: dense_scores.get(path, 0.0) for path in candidates})
        lexical_normalized = _minmax(
            {path: lexical_scores.get(path, 0.0) for path in candidates}
        )
        rrf_normalized = _minmax(rrf)
        query_terms = set(tokenize(query))
        record_by_path = {}
        for record in self.records:
            path = record["metadata"]["repository_path"]
            record_by_path.setdefault(path, record)
        results = []
        for path in candidates:
            metadata = record_by_path[path]["metadata"]
            title_terms = set(tokenize(metadata.get("title", "")))
            title_overlap = len(query_terms & title_terms) / max(1, len(title_terms))
            dense_weight, lexical_weight, rrf_weight, title_weight = self.weights
            score = (
                dense_weight * dense_normalized[path]
                + lexical_weight * lexical_normalized[path]
                + rrf_weight * rrf_normalized.get(path, 0.0)
                + title_weight * title_overlap
            )
            results.append(
                HybridHit(
                    document_id=record_by_path[path]["document_id"],
                    repository_path=path,
                    title=metadata["title"],
                    department=metadata["department"],
                    source_url=metadata["source_url"],
                    score=score,
                    dense_score=dense_scores.get(path, 0.0),
                    lexical_score=lexical_scores.get(path, 0.0),
                    rrf_score=rrf.get(path, 0.0),
                    title_overlap=title_overlap,
                )
            )
        return sorted(results, key=lambda hit: (-hit.score, hit.repository_path))[:top_k]

