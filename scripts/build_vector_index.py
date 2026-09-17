#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.embeddings import MODEL_NAME, SentenceTransformerEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed corpus chunks and persist a cosine index")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/processed/corpus-v1/chunks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/vector-index-v1")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in args.chunks.read_text().splitlines() if line.strip()
    ]
    embedder = SentenceTransformerEmbedder(args.model)
    embeddings = embedder.encode([record["text"] for record in records], args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", embeddings)
    with (args.output / "records.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_license": "Apache-2.0",
        "metric": "cosine_similarity",
        "index_type": "exact_numpy",
        "embedding_dimension": int(embeddings.shape[1]),
        "vector_count": int(embeddings.shape[0]),
        "normalized": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
