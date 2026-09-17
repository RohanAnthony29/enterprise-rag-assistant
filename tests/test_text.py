import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.models import Document
from enterprise_rag.text import chunk_document, clean_markup, infer_title, split_front_matter


class TextProcessingTests(unittest.TestCase):
    def test_front_matter_and_title(self):
        metadata, body = split_front_matter("---\ntitle: 'Security Policy'\n---\n# Ignored\nText")
        self.assertEqual(metadata["title"], "Security Policy")
        self.assertEqual(infer_title(metadata, body, "policy.md"), "Security Policy")

    def test_clean_markup(self):
        value = clean_markup("# Access\nRead the [policy](https://example.com). <b>Required</b>")
        self.assertEqual(value, "Access Read the policy. Required")

    def test_chunks_overlap_and_keep_metadata(self):
        document = Document(
            document_id="doc-1",
            title="Policy",
            text=" ".join(f"word-{index}" for index in range(12)),
            department="security",
            document_type="policy",
            source="test",
            source_url="https://example.com",
            repository_path="policy.md",
            source_revision="abc",
            retrieved_at_utc="2026-01-01T00:00:00Z",
        )
        chunks = chunk_document(document, size=5, overlap=2)
        self.assertEqual([chunk.word_count for chunk in chunks], [5, 5, 5, 3])
        self.assertEqual(chunks[0].text.split()[-2:], chunks[1].text.split()[:2])
        self.assertEqual(chunks[0].metadata["department"], "security")
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), 4)

    def test_invalid_chunk_configuration(self):
        document = Document("1", "t", "text", "d", "p", "s", "u", "path", "r", "now")
        with self.assertRaises(ValueError):
            chunk_document(document, size=5, overlap=5)


if __name__ == "__main__":
    unittest.main()
