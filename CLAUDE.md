# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system for querying markdown documentation. It chunks and embeds docs via Mistral AI, indexes them in Vespa, and serves a FastAPI + Next.js chat interface.

## Commands

### Python Backend

Uses `uv` as the package manager with Python 3.13.

```bash
uv run python -m backend.feed                              # Run full indexing pipeline
uv run python -m backend.feed --include_folders="agents"  # Index specific folders only
uv run python -m backend.feed --exclude_folders="old"     # Exclude folders

uv run python -m backend.rag "your question"              # CLI query (hybrid mode)
uv run python -m backend.rag "question" --top 10 --mode semantic

uv run python -m backend.server                           # Start FastAPI on http://0.0.0.0:8000
```

### Frontend

```bash
cd chat_interface
pnpm install
pnpm dev     # Starts Next.js on http://localhost:3000
pnpm build
pnpm lint
```

### Docker (Vespa + Backend)

```bash
docker-compose up   # Vespa on :8080/:19071, backend on :5000
```

## Architecture

### Data Flow

```
platform-docs-public/public/**/*.md
    → chunker.py (hierarchical splitting)
    → embedder.py (Mistral mistral-embed-2312, 1024-dim, batch=32)
    → vespa_utils.py (feed_all to Vespa)

User query (chat_interface → server.py /rag)
    → rag.py: embed query → search Vespa → Mistral LLM → answer
```

### Key Components

- **[backend/feed.py](backend/feed.py)**: Indexing pipeline orchestrator. Discovers MD files, chunks, embeds, deploys Vespa schema if needed, feeds docs.
- **[backend/chunker.py](backend/chunker.py)**: Recursive hierarchical splitting (H2→H3→H4→paragraphs→sentences). Protects code blocks from splitting. Min 50 / max 800 tokens, hard cap 8192. Merges undersized chunks.
- **[backend/embedder.py](backend/embedder.py)**: Mistral embedding with batch size 32, retry logic (3 attempts, 2s delay).
- **[backend/vespa_utils.py](backend/vespa_utils.py)**: Deploy schema, feed documents, query with `semantic` or `hybrid` ranking.
- **[backend/rag.py](backend/rag.py)**: Embeds query, retrieves top-k from Vespa, calls `mistral-large-latest` with context.
- **[backend/server.py](backend/server.py)**: FastAPI exposing `GET /health` and `POST /rag` (params: `query`, `top=5`, `mode=hybrid`). CORS allows localhost:3000.
- **[extract_all_md_files.py](extract_all_md_files.py)**: Finds `.md`/`.mdx` in `platform-docs-public/public/` with include/exclude folder filters.
- **[evaluation/evaluate.py](evaluation/evaluate.py)**: RAG evaluation pipeline (dataset generation, recall, faithfulness, answer relevancy, completeness).

### Vespa Schema ([my-vespa-app/schemas/doc.sd](my-vespa-app/schemas/doc.sd))

Fields: `id`, `source_file`, `heading`, `body` (BM25-indexed), `embedding` (tensor<float>[1024], angular).

Rank profiles:
- `semantic`: pure cosine similarity
- `hybrid`: 0.7 × semantic + 0.3 × BM25

### Environment

Requires `MISTRAL_API_KEY` in `.env`.
