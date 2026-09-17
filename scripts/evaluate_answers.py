#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.api import build_assistant
from enterprise_rag.lexical import tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounded answer quality")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/eval/answer_queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reports/answer-evaluation.json")
    args = parser.parse_args()
    assistant = build_assistant()
    cases = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    rows = []
    for case in cases:
        result = assistant.answer(case["question"], top_k=5, department=case.get("department"))
        answer_terms = set(tokenize(result.answer))
        expected = set(token.casefold() for token in case["expected_terms"])
        citation_ids = {int(value) for value in re.findall(r"\[(\d+)\]", result.answer)}
        valid_ids = {item["id"] for item in result.citations}
        rows.append(
            {
                "id": case["id"],
                "keyword_recall": len(expected & answer_terms) / max(1, len(expected)),
                "citation_present": bool(citation_ids),
                "citation_valid": bool(citation_ids) and citation_ids <= valid_ids,
                "source_retrieved": case["expected_path"]
                in {item["repository_path"] for item in result.citations},
                "latency_ms": result.total_ms,
            }
        )
    latencies = [row["latency_ms"] for row in rows]
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    summary = {
        "queries": len(rows),
        "keyword_recall": statistics.fmean(row["keyword_recall"] for row in rows),
        "citation_coverage": statistics.fmean(row["citation_present"] for row in rows),
        "citation_validity": statistics.fmean(row["citation_valid"] for row in rows),
        "expected_source_recall": statistics.fmean(row["source_retrieved"] for row in rows),
        "cold_start_ms": latencies[0],
        "warm_latency_ms_mean": statistics.fmean(warm_latencies),
        "warm_latency_ms_p95": sorted(warm_latencies)[
            max(0, math.ceil(0.95 * len(warm_latencies)) - 1)
        ],
    }
    payload = {"summary": summary, "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
