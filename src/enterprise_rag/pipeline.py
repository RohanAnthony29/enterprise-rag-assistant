from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .gitlab_source import GitLabHandbookSource
from .models import Document
from .text import chunk_document, clean_markup, infer_title, split_front_matter


def build_corpus(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    source = GitLabHandbookSource(config["project"], config["ref"])
    files = source.list_files(
        config["department_paths"],
        tuple(config["allowed_extensions"]),
        int(config["max_documents"]),
    )
    retrieved_at = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    document_count = chunk_count = 0
    departments: dict[str, int] = {}
    with documents_path.open("w", encoding="utf-8") as documents_file, chunks_path.open(
        "w", encoding="utf-8"
    ) as chunks_file:
        for source_file in files:
            raw = source.read_file(source_file.path)
            front_matter, body = split_front_matter(raw)
            text = clean_markup(body)
            if len(text.split()) < 50:
                continue
            document_id = hashlib.sha256(source_file.path.encode("utf-8")).hexdigest()[:20]
            document = Document(
                document_id=document_id,
                title=infer_title(front_matter, body, source_file.path),
                text=text,
                department=source_file.department,
                document_type="handbook_page",
                source=config["source"],
                source_url=source.web_url(source_file.path),
                repository_path=source_file.path,
                source_revision=source_file.revision,
                retrieved_at_utc=retrieved_at,
            )
            documents_file.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
            document_count += 1
            departments[document.department] = departments.get(document.department, 0) + 1
            for chunk in chunk_document(
                document,
                int(config["chunk_size_words"]),
                int(config["chunk_overlap_words"]),
            ):
                chunks_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                chunk_count += 1
    manifest = {
        "created_at_utc": retrieved_at,
        "source": config["source"],
        "source_project": config["project"],
        "source_ref": config["ref"],
        "license": "MIT",
        "license_url": f"https://gitlab.com/{config['project']}/-/blob/{config['ref']}/LICENSE",
        "document_count": document_count,
        "chunk_count": chunk_count,
        "departments": departments,
        "chunk_size_words": config["chunk_size_words"],
        "chunk_overlap_words": config["chunk_overlap_words"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

