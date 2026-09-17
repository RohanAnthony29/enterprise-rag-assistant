from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "where", "which", "who", "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS]


@dataclass(frozen=True)
class LexicalHit:
    record_index: int
    score: float


class BM25Index:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.records = records
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(self._search_text(record))) for record in records]
        self.lengths = [sum(values.values()) for values in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for values in self.term_frequencies:
            document_frequency.update(values.keys())
        total = len(records)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def _search_text(record: dict) -> str:
        metadata = record.get("metadata", {})
        return f"{metadata.get('title', '')} {record.get('text', '')}"

    def search(
        self, query: str, top_k: int = 10, department: str | None = None
    ) -> list[LexicalHit]:
        query_terms = tokenize(query)
        scores: list[LexicalHit] = []
        for index, frequencies in enumerate(self.term_frequencies):
            metadata = self.records[index].get("metadata", {})
            if department is not None and metadata.get("department") != department:
                continue
            length_norm = 1 - self.b + self.b * self.lengths[index] / max(
                self.average_length, 1e-12
            )
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1)
                        / (frequency + self.k1 * length_norm)
                    )
            if score > 0:
                scores.append(LexicalHit(index, score))
        return sorted(scores, key=lambda item: (-item.score, item.record_index))[:top_k]

