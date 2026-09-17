import unittest
from unittest.mock import patch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.gitlab_source import GitLabHandbookSource


class GitLabSourceTests(unittest.TestCase):
    def test_selects_supported_files_with_balanced_departments(self):
        source = GitLabHandbookSource("group/project")
        payloads = {
            "content/hr": [
                {"type": "blob", "path": "content/hr/a.md", "id": "1"},
                {"type": "blob", "path": "content/hr/b.png", "id": "2"},
            ],
            "content/it": [
                {"type": "blob", "path": "content/it/a.html", "id": "3"},
                {"type": "blob", "path": "content/it/b.md", "id": "4"},
            ],
        }

        def fake_json(_endpoint, parameters):
            return payloads[parameters["path"]], {"x-next-page": ""}

        with patch.object(source, "_json", side_effect=fake_json):
            files = source.list_files(
                {"hr": "content/hr", "it": "content/it"}, (".md", ".html"), 4
            )
        self.assertEqual([item.department for item in files], ["hr", "it", "it"])
        self.assertNotIn("content/hr/b.png", [item.path for item in files])


if __name__ == "__main__":
    unittest.main()
