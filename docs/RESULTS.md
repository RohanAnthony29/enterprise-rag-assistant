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
