# Retrieval baseline

Measured locally on 2026-09-17 using 767 chunks from 69 public GitLab Handbook
documents and 26 manually curated, source-grounded queries.

The index uses normalized 384-dimensional
`sentence-transformers/all-MiniLM-L6-v2` embeddings and exact cosine similarity.
Retrieved chunks are deduplicated into document rankings before evaluation.

| K | Recall@K | MRR@K | NDCG@K |
|---:|---:|---:|---:|
| 1 | 0.6731 | 0.6923 | 0.6923 |
| 3 | 0.9231 | 0.7949 | 0.8278 |
| 5 | 0.9231 | 0.7949 | 0.8278 |
| 10 | 0.9615 | 0.8013 | 0.8415 |

Exact index lookup averaged 0.20 ms with a measured p95 of 0.40 ms. This excludes
query-embedding time and reflects the current small local corpus.

The missed Recall@10 case was “What are the guidelines for workplace chat
communication?” The dense index preferred general communication and employee
engagement pages over the dedicated `communication/chat.md` document. This is a
good target for lexical+dense hybrid retrieval, metadata-aware scoring, or a
cross-encoder reranker.

The evaluation set is intentionally checked into the repository for review. It
is a development baseline authored from the available corpus, not an independent
blind test set. A later milestone should add paraphrased queries written without
viewing document titles and separate development and test query sets.

## Hybrid retrieval and reranking

The hybrid system combines dense cosine retrieval and BM25 lexical retrieval,
deduplicates chunk results into documents, applies reciprocal-rank fusion, and
reranks candidates using normalized dense similarity, BM25 relevance, RRF, and
query-to-title token overlap. An optional department filter is enforced by both
candidate sources before fusion.

| Metric | Dense baseline | Hybrid + reranking | Relative lift |
|---|---:|---:|---:|
| Recall@1 | 0.6731 | 0.8654 | +28.57% |
| MRR@1 | 0.6923 | 0.8846 | +27.78% |
| NDCG@1 | 0.6923 | 0.8846 | +27.78% |
| Recall@10 | 0.9615 | 1.0000 | +4.00% |
| MRR@10 | 0.8013 | 0.9295 | +16.00% |
| NDCG@10 | 0.8415 | 0.9473 | +12.57% |

Hybrid retrieval averaged 1.37 ms with a measured p95 of 1.61 ms, excluding
query embedding. The additional latency is small for this corpus and buys a
substantial improvement in first-result relevance. A small development-set
weight check confirmed the checked-in `0.50/0.30/0.10/0.10` dense, lexical, RRF,
and title configuration performed better than the tested title-heavier variants.

## Answer generation and service evaluation

The completed assistant selects the strongest chunk from each retrieved document
and supplies those contexts to a deterministic extractive generator or an
optional local Ollama model. Both paths return numbered citations mapped to the
original source URL. The extractive path is the reproducible default because it
has no paid dependency and does not transmit source text externally.

`scripts/evaluate_answers.py` measures expected-source retrieval, expected-keyword
coverage, citation coverage, citation validity, and end-to-end latency. These are
transparent automated proxies, not a substitute for human review. The service
benchmark in `scripts/benchmark_api.py` reports success rate, throughput, mean,
p50, p95, and p99 latency after one warm-up request.

Measured API latency is environment-dependent. The local smoke-test result below
is labeled with its workload and must not be represented as a production
service-level objective.

### Measured completion run

The final local extractive-answer evaluation used four representative queries
with department filters:

| Metric | Result |
|---|---:|
| Expected-source Recall@5 | 1.0000 |
| Citation coverage | 1.0000 |
| Citation validity | 1.0000 |
| Expected-keyword recall | 0.8750 |
| Warm answer latency, mean | 31.56 ms |
| Warm answer latency, p95 | 40.55 ms |

The API load smoke test issued 20 requests at concurrency 4 after model warm-up.
All requests succeeded, throughput was 66.12 requests/second, mean latency was
59.49 ms, and p95 latency was 80.94 ms. This run used the local extractive
generator and a 767-chunk index. The sample sizes are deliberately small and the
numbers should be re-measured on the eventual deployment target.

## Local LLM comparison

Measured on 2026-09-17 in the 2-core GitHub Codespace with Ollama. The baseline
used `qwen2.5:0.5b` and a generic context prompt. The optimized path used
`llama3.2:1b`, the same hybrid retrieval layer, and the citation-constrained
grounding prompt used by the FastAPI service.

The response-quality score was defined before the run as the unweighted mean of
expected-keyword recall, evidence-token grounding, citation coverage, and
citation validity.

| Metric | Qwen baseline | Optimized Llama RAG |
|---|---:|---:|
| Expected-keyword recall | 1.0000 | 1.0000 |
| Evidence-token grounding | 0.7845 | 0.8422 |
| Citation coverage | 0.0000 | 0.2500 |
| Citation validity | 0.0000 | 0.2500 |
| Composite response quality | 0.4461 | 0.5856 |
| Mean generation latency | 29.57 s | 69.58 s |

The optimized system improved the declared composite score by **31.25%**. The
small development set and automated token-overlap metric make this directional
evidence, not a universal quality claim. Citation compliance also remained a
clear failure mode for this very small model. The Llama generator was separately
served through `POST /v1/answer`; a warm CPU request returned successfully with
three retrieved citations in 22.32 seconds.
