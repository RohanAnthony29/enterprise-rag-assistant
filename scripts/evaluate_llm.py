#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enterprise_rag.api import build_assistant
from enterprise_rag.generation import OllamaGenerator
from enterprise_rag.lexical import tokenize


def score_answer(answer: str, expected_terms: list[str], contexts) -> dict:
    answer_terms = set(tokenize(answer))
    evidence_terms = set(tokenize(" ".join(context.text for context in contexts)))
    expected = {term.casefold() for term in expected_terms}
    citations = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    valid_ids = {context.rank for context in contexts}
    keyword_recall = len(answer_terms & expected) / max(1, len(expected))
    grounded_token_ratio = len(answer_terms & evidence_terms) / max(1, len(answer_terms))
    citation_coverage = float(bool(citations))
    citation_validity = float(bool(citations) and citations <= valid_ids)
    quality = statistics.fmean(
        [keyword_recall, grounded_token_ratio, citation_coverage, citation_validity]
    )
    return {
        "keyword_recall": keyword_recall,
        "grounded_token_ratio": grounded_token_ratio,
        "citation_coverage": citation_coverage,
        "citation_validity": citation_validity,
        "response_quality": quality,
    }


def evaluate(generator, cases, assistant) -> tuple[dict, list[dict]]:
    rows = []
    for case in cases:
        contexts = assistant.retrieve(case["question"], top_k=5, department=case.get("department"))
        started = time.perf_counter()
        answer = generator.generate(case["question"], contexts)
        latency_ms = (time.perf_counter() - started) * 1000
        metrics = score_answer(answer, case["expected_terms"], contexts)
        rows.append(
            {
                "id": case["id"],
                "answer": answer,
                "latency_ms": latency_ms,
                **metrics,
            }
        )
    summary = {
        key: statistics.fmean(row[key] for row in rows)
        for key in (
            "keyword_recall",
            "grounded_token_ratio",
            "citation_coverage",
            "citation_validity",
            "response_quality",
            "latency_ms",
        )
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized local LLM RAG")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/eval/answer_queries.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reports/llm-evaluation.json")
    parser.add_argument("--baseline-model", default="qwen2.5:0.5b")
    parser.add_argument("--optimized-model", default="llama3.2:1b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    assistant = build_assistant()
    baseline = OllamaGenerator(args.baseline_model, args.ollama_url, 180, require_citations=False)
    optimized = OllamaGenerator(args.optimized_model, args.ollama_url, 180, require_citations=True)
    baseline_summary, baseline_rows = evaluate(baseline, cases, assistant)
    optimized_summary, optimized_rows = evaluate(optimized, cases, assistant)
    baseline_score = baseline_summary["response_quality"]
    lift = (optimized_summary["response_quality"] - baseline_score) / max(baseline_score, 1e-12)
    report = {
        "methodology": {
            "quality_formula": "mean(keyword_recall, grounded_token_ratio, citation_coverage, citation_validity)",
            "queries": len(cases),
            "baseline_model": args.baseline_model,
            "optimized_model": args.optimized_model,
        },
        "baseline": {"summary": baseline_summary, "cases": baseline_rows},
        "optimized": {"summary": optimized_summary, "cases": optimized_rows},
        "relative_response_quality_lift": lift,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "relative_response_quality_lift": lift,
    }, indent=2))


if __name__ == "__main__":
    main()
