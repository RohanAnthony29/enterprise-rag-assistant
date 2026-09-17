#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def request_once(url: str, question: str) -> tuple[float, int]:
    payload = json.dumps({"question": question, "top_k": 5}).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/answer",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
        return (time.perf_counter() - started) * 1000, response.status


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent API latency benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    question = "How should open-source dependencies be secured?"
    request_once(args.url, question)  # warm model and index
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request_once, args.url, question) for _ in range(args.requests)]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started
    latency = [item[0] for item in results]
    report = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "success_rate": sum(status == 200 for _, status in results) / len(results),
        "throughput_rps": len(results) / elapsed,
        "latency_ms": {
            "mean": statistics.fmean(latency),
            "p50": percentile(latency, 0.50),
            "p95": percentile(latency, 0.95),
            "p99": percentile(latency, 0.99),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
