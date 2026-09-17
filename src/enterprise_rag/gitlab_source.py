from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass


API_ROOT = "https://gitlab.com/api/v4"


@dataclass(frozen=True)
class SourceFile:
    path: str
    revision: str
    department: str


class GitLabHandbookSource:
    def __init__(self, project: str, ref: str = "main", timeout_seconds: int = 30):
        self.project = project
        self.ref = ref
        self.timeout_seconds = timeout_seconds
        self.encoded_project = urllib.parse.quote(project, safe="")

    def _json(self, endpoint: str, parameters: dict[str, str | int]) -> tuple[object, dict]:
        query = urllib.parse.urlencode(parameters)
        request = urllib.request.Request(
            f"{API_ROOT}/projects/{self.encoded_project}/{endpoint}?{query}",
            headers={"User-Agent": "enterprise-rag-assistant/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response), {
                key.lower(): value for key, value in response.headers.items()
            }

    def list_files(
        self,
        department_paths: dict[str, str],
        extensions: tuple[str, ...],
        max_documents: int,
    ) -> list[SourceFile]:
        if max_documents <= 0:
            return []
        per_department = max(1, max_documents // len(department_paths))
        selected: list[SourceFile] = []
        for department, path in department_paths.items():
            entries: list[dict] = []
            page = 1
            while True:
                payload, headers = self._json(
                    "repository/tree",
                    {
                        "path": path,
                        "ref": self.ref,
                        "recursive": "true",
                        "per_page": 100,
                        "page": page,
                    },
                )
                entries.extend(payload)
                next_page = headers.get("x-next-page", "")
                if not next_page:
                    break
                page = int(next_page)
            files = sorted(
                (
                    SourceFile(entry["path"], entry["id"], department)
                    for entry in entries
                    if entry["type"] == "blob"
                    and entry["path"].lower().endswith(extensions)
                ),
                key=lambda item: item.path,
            )
            selected.extend(files[:per_department])
        return selected[:max_documents]

    def read_file(self, path: str) -> str:
        encoded_path = urllib.parse.quote(path, safe="")
        request = urllib.request.Request(
            f"{API_ROOT}/projects/{self.encoded_project}/repository/files/"
            f"{encoded_path}/raw?{urllib.parse.urlencode({'ref': self.ref})}",
            headers={"User-Agent": "enterprise-rag-assistant/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")

    def web_url(self, path: str) -> str:
        return f"https://gitlab.com/{self.project}/-/blob/{self.ref}/{path}"
