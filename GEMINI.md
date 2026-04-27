# GEMINI.md - Project Context

## Project Overview
**embed-reranker** is a lightweight, Japanese-specialized API server that provides both Embedding and Reranking capabilities in a single process. It is built using FastAPI and optimized for RAG (Retrieval-Augmented Generation) workflows.

### Key Technologies
- **Language:** Python 3.11+ (managed by `uv`)
- **Web Framework:** FastAPI + Uvicorn
- **AI/ML:** PyTorch (MPS/Apple Silicon supported), Transformers, Sentence-Transformers
- **Models:**
  - **Embedding:** `cl-nagoya/ruri-v3-70m` (Mean Pooling + L2 Normalization)
  - **Reranker:** `cl-nagoya/ruri-v3-reranker-310m` (CrossEncoder)

### Architecture
The server exposes an OpenAI-compatible `/v1/embeddings` endpoint and a specialized `/v1/rerank` (or `/rerank`) endpoint. It automatically detects and utilizes Metal Performance Shaders (MPS) on Apple Silicon for accelerated inference.

## Building and Running

### Prerequisites
- Python 3.11 or higher
- `uv` (recommended for dependency management)

### Development Setup
```bash
# Install dependencies using uv
uv sync
```

### Running the Server
The server runs on port `1235` by default.

```bash
# Start via the provided shell script
./run_ruri_server.sh

# Or directly via uvicorn
uv run uvicorn ruri_embed_rerank_server:app --host 127.0.0.1 --port 1235
```

### Automated Startup (macOS)
The project is configured to auto-start via `launchd` on macOS.
- **plist:** `~/Library/LaunchAgents/com.norihito.embed-reranker.plist`
- **Logs:** `~/Library/Logs/com.norihito.embed-reranker.log`

## Development Conventions

### Coding Style
- **Type Hinting:** Extensive use of Pydantic models for request/response validation.
- **Inference:** Inference is performed with `torch.no_grad()` for efficiency.
- **Portability:** The code explicitly checks for `mps` availability and falls back to `cpu`.

### API Specifications
- **Health Check:** `GET /health` returns status and model info.
- **Embedding:** `POST /v1/embeddings` (OpenAI-compatible format).
- **Rerank:** `POST /v1/rerank` or `POST /rerank` (Accepts `query`, `documents`, and `top_k`).

## Key Files
- `ruri_embed_rerank_server.py`: The main FastAPI implementation.
- `run_ruri_server.sh`: Startup script using `uv`.
- `pyproject.toml`: Project metadata and dependencies.
- `AUTO_STARTUP_SUMMARY.md`: Documentation for the `launchd` setup.
