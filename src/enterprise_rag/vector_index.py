from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import normalize_rows


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict


class NumpyVectorIndex:
    """Persisted exact cosine index suitable for the initial corpus baseline."""

    def __init__(self, embeddings: np.ndarray, records: list[dict]):
        if embeddings.ndim != 2 or len(embeddings) != len(records):
            raise ValueError("embeddings and records must have matching rows")
        self.embeddings = normalize_rows(embeddings)
        self.records = records

    @classmethod
    def load(cls, index_dir: Path) -> "NumpyVectorIndex":
        embeddings = np.load(index_dir / "embeddings.npy")
        records = [
            json.loads(line)
            for line in (index_dir / "records.jsonl").read_text().splitlines()
            if line.strip()
        ]
        return cls(embeddings, records)

    def search(
        self, query_vector: np.ndarray, top_k: int = 10, department: str | None = None
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        query = normalize_rows(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query and index dimensions do not match")
        allowed = np.array(
            [
                department is None or record["metadata"].get("department") == department
                for record in self.records
            ],
            dtype=bool,
        )
        indexes = np.flatnonzero(allowed)
        if not len(indexes):
            return []
        scores = self.embeddings[indexes] @ query
        count = min(top_k, len(indexes))
        ranked_local = np.argpartition(-scores, count - 1)[:count]
        ranked_local = ranked_local[np.argsort(-scores[ranked_local], kind="stable")]
        hits = []
        for local_index in ranked_local:
            record = self.records[int(indexes[local_index])]
            hits.append(
                SearchHit(
                    chunk_id=record["chunk_id"],
                    document_id=record["document_id"],
                    score=float(scores[local_index]),
                    text=record["text"],
                    metadata=record["metadata"],
                )
            )
        return hits
