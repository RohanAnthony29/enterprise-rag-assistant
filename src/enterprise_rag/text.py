from __future__ import annotations

import hashlib
import html
import re

from .models import Chunk, Document


FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HTML_TAG = re.compile(r"<[^>]+>")
MARKDOWN_LINK = re.compile(r"!?\[([^]]*)\]\([^)]+\)")


def split_front_matter(content: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(content)
    if not match:
        return {}, content
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        if value:
            metadata[key.strip().lower()] = value
    return metadata, content[match.end() :]


def clean_markup(content: str) -> str:
    content = re.sub(r"\{\{<.*?>\}\}", " ", content, flags=re.DOTALL)
    content = MARKDOWN_LINK.sub(lambda match: match.group(1), content)
    content = re.sub(r"```.*?```", " ", content, flags=re.DOTALL)
    content = HTML_TAG.sub(" ", content)
    content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"[*_`>|]", " ", content)
    content = html.unescape(content)
    return re.sub(r"\s+", " ", content).strip()


def infer_title(front_matter: dict[str, str], body: str, path: str) -> str:
    if front_matter.get("title"):
        return front_matter["title"]
    heading = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if heading:
        return heading.group(1).strip()
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("-", " ").title()


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be in [0, size)")
    words = document.text.split()
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        text = " ".join(words[start:end])
        digest = hashlib.sha256(
            f"{document.document_id}:{len(chunks)}:{text}".encode("utf-8")
        ).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{digest}",
                document_id=document.document_id,
                text=text,
                chunk_index=len(chunks),
                word_count=end - start,
                metadata={
                    "title": document.title,
                    "department": document.department,
                    "document_type": document.document_type,
                    "source": document.source,
                    "source_url": document.source_url,
                    "repository_path": document.repository_path,
                    "source_revision": document.source_revision,
                    "access_level": document.access_level,
                },
            )
        )
        if end == len(words):
            break
        start = end - overlap
    return chunks
