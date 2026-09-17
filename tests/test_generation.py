import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.generation import Context, ExtractiveGenerator


class ExtractiveGenerationTests(unittest.TestCase):
    def test_answer_is_grounded_and_cited(self):
        contexts = [
            Context(1, "a", "Access", "Identity access reviews happen quarterly. Owners remove stale access.", "security/access.md", "https://example.com/a", "security", 0.9)
        ]
        answer = ExtractiveGenerator(max_sentences=1).generate("When are access reviews?", contexts)
        self.assertIn("quarterly", answer)
        self.assertIn("[1]", answer)

    def test_no_evidence_is_explicit(self):
        answer = ExtractiveGenerator().generate("unknown", [])
        self.assertIn("could not find enough evidence", answer)


if __name__ == "__main__":
    unittest.main()
