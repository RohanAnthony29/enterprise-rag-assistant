#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.embeddings import MODEL_NAME, SentenceTransformerEmbedder
from enterprise_rag.hybrid import HybridRetriever
from enterprise_rag.vector_index import NumpyVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the enterprise knowledge index")
    parser.add_argument("query")
    parser.add_argument("--department")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts/vector-index-v1")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    index = NumpyVectorIndex.load(args.index)
    embedder = SentenceTransformerEmbedder(args.model)
    vector = embedder.encode([args.query])[0]
    hits = HybridRetriever(index).search(
        args.query,
        vector,
        top_k=args.top_k,
        department=args.department,
    )
    print(
        json.dumps(
            [
                {
                    "rank": rank,
                    "title": hit.title,
                    "department": hit.department,
                    "source_url": hit.source_url,
                    "score": hit.score,
                    "signals": {
                        "dense": hit.dense_score,
                        "lexical": hit.lexical_score,
                        "rrf": hit.rrf_score,
                        "title_overlap": hit.title_overlap,
                    },
                }
                for rank, hit in enumerate(hits, start=1)
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

