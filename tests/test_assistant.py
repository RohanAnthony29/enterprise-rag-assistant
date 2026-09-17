import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.assistant import KnowledgeAssistant
from enterprise_rag.generation import ExtractiveGenerator
from enterprise_rag.vector_index import NumpyVectorIndex


class FakeEmbedder:
    model_name = "fake"

    def encode(self, texts, batch_size=32):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class AssistantTests(unittest.TestCase):
    def test_answer_contains_source_metadata(self):
        records = [{
            "chunk_id": "c1", "document_id": "d1", "text": "Access reviews happen quarterly.",
            "metadata": {"title": "Access reviews", "department": "security", "repository_path": "security/access.md", "source_url": "https://example.com/access"},
        }]
        assistant = KnowledgeAssistant(
            NumpyVectorIndex(np.array([[1.0, 0.0]], dtype=np.float32), records),
            FakeEmbedder(), ExtractiveGenerator(),
        )
        result = assistant.answer("When do access reviews happen?", department="security")
        self.assertEqual(result.citations[0]["source_url"], "https://example.com/access")
        self.assertIn("[1]", result.answer)
        self.assertGreaterEqual(result.total_ms, 0)


if __name__ == "__main__":
    unittest.main()
