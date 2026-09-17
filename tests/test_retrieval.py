import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from enterprise_rag.vector_index import NumpyVectorIndex


class RetrievalTests(unittest.TestCase):
    def test_metrics_reward_relevant_ranking(self):
        ranked = ["other", "relevant", "second"]
        relevant = {"relevant", "second"}
        self.assertEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(reciprocal_rank_at_k(ranked, relevant, 3), 0.5)
        self.assertGreater(ndcg_at_k(ranked, relevant, 3), 0.6)

    def test_cosine_search_and_metadata_filter(self):
        embeddings = np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32)
        records = [
            {"chunk_id": "a", "document_id": "1", "text": "a", "metadata": {"department": "hr"}},
            {"chunk_id": "b", "document_id": "2", "text": "b", "metadata": {"department": "it"}},
            {"chunk_id": "c", "document_id": "3", "text": "c", "metadata": {"department": "hr"}},
        ]
        index = NumpyVectorIndex(embeddings, records)
        self.assertEqual(index.search(np.array([1, 0]), 1)[0].chunk_id, "a")
        self.assertEqual(
            index.search(np.array([1, 0]), 1, department="it")[0].chunk_id,
            "b",
        )


if __name__ == "__main__":
    unittest.main()

