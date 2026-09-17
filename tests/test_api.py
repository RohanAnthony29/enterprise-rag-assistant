import os
import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag import api
from enterprise_rag.assistant import KnowledgeAssistant
from enterprise_rag.generation import ExtractiveGenerator
from enterprise_rag.vector_index import NumpyVectorIndex


class FakeEmbedder:
    model_name = "fake"

    def encode(self, texts, batch_size=32):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = [{
            "chunk_id": "c1", "document_id": "d1", "text": "Access reviews happen quarterly.",
            "metadata": {"title": "Access reviews", "department": "security", "repository_path": "security/access.md", "source_url": "https://example.com/access"},
        }]
        api._assistant = KnowledgeAssistant(
            NumpyVectorIndex(np.array([[1.0, 0.0]], dtype=np.float32), records),
            FakeEmbedder(), ExtractiveGenerator(),
        )
        cls.client = TestClient(api.app)

    def tearDown(self):
        os.environ.pop("RAG_API_KEY", None)

    def test_health_and_answer(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.post(
            "/v1/answer",
            json={"question": "When do access reviews happen?", "department": "security"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"][0]["repository_path"], "security/access.md")

    def test_request_validation(self):
        response = self.client.post("/v1/answer", json={"question": "x", "top_k": 100})
        self.assertEqual(response.status_code, 422)

    def test_optional_api_key(self):
        os.environ["RAG_API_KEY"] = "secret"
        self.assertEqual(self.client.get("/ready").status_code, 401)
        self.assertEqual(
            self.client.get("/ready", headers={"X-API-Key": "secret"}).status_code,
            200,
        )

    def test_metrics_are_exposed(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("rag_requests_total", response.text)


if __name__ == "__main__":
    unittest.main()
