#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.embeddings import MODEL_NAME, SentenceTransformerEmbedder
from enterprise_rag.hybrid import HybridRetriever
from enterprise_rag.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from enterprise_rag.vector_index import NumpyVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense+BM25 hybrid retrieval")
    parser.add_argument("--index", type=Path, default=ROOT / "artifacts/vector-index-v1")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/eval/retrieval_queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reports/hybrid_retrieval.json")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    queries = [
        json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()
    ]
    index = NumpyVectorIndex.load(args.index)
    retriever = HybridRetriever(index)
    embedder = SentenceTransformerEmbedder(args.model)
    vectors = embedder.encode([item["query"] for item in queries])
    cutoffs = (1, 3, 5, 10)
    per_query, latencies = [], []
    for item, vector in zip(queries, vectors):
        started = time.perf_counter()
        hits = retriever.search(item["query"], vector, top_k=10)
        latencies.append((time.perf_counter() - started) * 1000)
        ranked = [hit.repository_path for hit in hits]
        relevant = set(item["relevant_paths"])
        metrics = {}
        for cutoff in cutoffs:
            metrics[f"recall@{cutoff}"] = recall_at_k(ranked, relevant, cutoff)
            metrics[f"mrr@{cutoff}"] = reciprocal_rank_at_k(ranked, relevant, cutoff)
            metrics[f"ndcg@{cutoff}"] = ndcg_at_k(ranked, relevant, cutoff)
        per_query.append(
            {
                **item,
                "metrics": metrics,
                "top_results": [
                    {
                        "path": hit.repository_path,
                        "score": hit.score,
                        "dense_score": hit.dense_score,
                        "lexical_score": hit.lexical_score,
                        "title_overlap": hit.title_overlap,
                    }
                    for hit in hits
                ],
            }
        )
    metrics = {
        name: statistics.fmean(item["metrics"][name] for item in per_query)
        for name in per_query[0]["metrics"]
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "retriever": "dense_bm25_rrf_feature_reranker",
        "queries": len(queries),
        "metrics": metrics,
        "retrieval_latency_ms": {
            "mean": statistics.fmean(latencies),
            "p95": sorted(latencies)[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
        },
        "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()

