# Enterprise RAG Assistant

A production-style Generative AI knowledge assistant built with retrieval-augmented
generation. The project uses a bounded subset of the public GitLab Handbook as a
realistic enterprise corpus and preserves document-level metadata for filtering,
versioning, and citations.

## Current milestone: ingestion and chunking

The first pipeline:

1. Discovers Markdown and HTML handbook pages through the public GitLab API.
2. Selects a balanced subset across People, IT, Engineering, Security, and
   Communication departments.
3. Extracts front matter, cleans markup, and assigns deterministic document IDs.
4. Produces overlapping chunks with inherited metadata and stable chunk IDs.
5. Writes a version manifest with source, license, counts, and chunk settings.

Build the initial corpus:

```bash
make corpus
```

Outputs are written to `data/processed/corpus-v1/` and intentionally excluded
from Git. The source content belongs to GitLab and is used under the repository's
MIT license; every document retains its original source URL and revision.

Run tests:

```bash
make test
```

## Metadata contract

Each chunk includes `document_id`, `chunk_id`, title, department, document type,
source URL, repository revision, access level, chunk index, and word count. These
fields will support vector-search filtering and citation generation in the next
milestone.

## Embeddings and retrieval baseline

Create an isolated environment and install the local retrieval dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-retrieval.txt
```

Generate normalized 384-dimensional embeddings with the Apache-2.0 licensed
`sentence-transformers/all-MiniLM-L6-v2` model and persist the exact cosine
index:

```bash
make index
make evaluate
```

Evaluation uses a checked-in, human-readable relevance set. Chunk hits are
deduplicated to document rankings before macro-averaged Recall@K, MRR@K, and
NDCG@K are calculated. Generated embeddings and reports are excluded from Git.

Current 26-query baseline:

- Recall@10: `0.9615`
- MRR@10: `0.8013`
- NDCG@10: `0.8415`
- Exact vector-search latency: `0.20 ms` mean and `0.40 ms` p95

See [the retrieval results](docs/RESULTS.md) for the complete table, evaluation
limitations, and the first diagnosed failure case.

## Hybrid and metadata-aware search

The second retrieval stage combines dense cosine similarity with an in-memory
BM25 index. Candidate documents are fused with reciprocal rank fusion and
reranked using dense, lexical, RRF, and title-overlap signals. Both dense and
lexical retrieval enforce an optional department filter before fusion.

```bash
make evaluate-hybrid
make search QUERY="How should open-source dependencies be secured?"
make search QUERY="How is identity access managed?" DEPARTMENT=security
```

On the same 26-query development set, hybrid reranking reached Recall@10 of
`1.0000`, MRR@10 of `0.9295`, and NDCG@10 of `0.9473`. Compared with the dense
baseline, this is a `12.57%` NDCG@10 improvement and a `27.78%` NDCG@1
improvement. Hybrid lookup measured `1.37 ms` mean and `1.61 ms` p95, excluding
query embedding.
