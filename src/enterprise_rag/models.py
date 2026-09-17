from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    department: str
    document_type: str
    source: str
    source_url: str
    repository_path: str
    source_revision: str
    retrieved_at_utc: str
    access_level: str = "public"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    word_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

