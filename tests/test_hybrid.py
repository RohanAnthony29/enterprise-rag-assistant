import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.hybrid import HybridRetriever
from enterprise_rag.lexical import BM25Index
from enterprise_rag.vector_index import NumpyVectorIndex


def record(chunk_id, document_id, title, department, path, text):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "metadata": {
            "title": title,
            "department": department,
            "repository_path": path,
            "source_url": f"https://example.com/{path}",
        },
    }


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("1", "a", "General Communication", "people", "general.md", "company communication rules"),
            record("2", "b", "Workplace Chat", "communication", "chat.md", "chat channels and messaging etiquette"),
            record("3", "c", "Security Policy", "security", "security.md", "identity access controls"),
        ]

    def test_bm25_rewards_matching_terms(self):
        hits = BM25Index(self.records).search("workplace chat messaging", top_k=2)
        self.assertEqual(hits[0].record_index, 1)

    def test_metadata_filter_is_enforced(self):
        vectors = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        retriever = HybridRetriever(NumpyVectorIndex(vectors, self.records))
        hits = retriever.search(
            "identity access", np.array([1, 0]), top_k=3, department="security"
        )
        self.assertEqual([hit.repository_path for hit in hits], ["security.md"])

    def test_reranker_can_promote_exact_title_match(self):
        vectors = np.array([[1, 0], [0.95, 0.05], [0, 1]], dtype=np.float32)
        retriever = HybridRetriever(NumpyVectorIndex(vectors, self.records))
        hits = retriever.search("workplace chat", np.array([1, 0]), top_k=2)
        self.assertEqual(hits[0].repository_path, "chat.md")


if __name__ == "__main__":
    unittest.main()

