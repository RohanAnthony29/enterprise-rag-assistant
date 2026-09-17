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

