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

