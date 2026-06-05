# GEMINI.md - Project Context

## Project Overview
**embed-reranker** is a lightweight, Japanese-specialized API server that provides both Embedding and Reranking capabilities in a single process. It is optimized for RAG (Retrieval-Augmented Generation) workflows and supports both standard PyTorch (MPS) and Apple Silicon native MLX backends.

### Key Technologies
- **Language:** Python 3.13+ (managed by `uv`)
- **Web Framework:** FastAPI + Uvicorn
- **AI/ML Backends:**
  - **MLX (Primary):** Native Apple Silicon optimization. **Current focus.**
  - **PyTorch (Legacy/Compatibility):** MPS/CPU support. (Note: The Ruri-based implementation is currently **frozen**.)
- **Models (MLX Backend):**
  - **Embedding:** `mlx-community/embedding-gemma-300m-bf16`, `mlx-community/bge-m3-mlx-fp16`
  - **Reranker:** `mlx-community/Qwen3-Reranker-0.6B-mxfp8`, `mlx-community/japanese-bge-reranker-v2-m3`
- **Models (Ruri Backend):**
  - **Embedding:** `cl-nagoya/ruri-v3-70m`
  - **Reranker:** `cl-nagoya/ruri-v3-reranker-310m`

### Architecture
The server exposes an OpenAI-compatible `/v1/embeddings` endpoint and a specialized `/v1/rerank` (or `/rerank`) endpoint. The MLX backend provides significantly lower latency and memory footprint on Mac hardware.

## Building and Running

### Prerequisites
- Python 3.13 or higher
- `uv` (required for dependency management)

### Development Setup
```bash
# Install dependencies using uv
uv sync
```

### Running the Server
The server runs on port `1235` by default.

#### MLX Version (Recommended for Mac)
```bash
# Start via the MLX shell script
./run_mlx_server.sh

# Or directly via uv
uv run uvicorn mlx_embed_rerank_server:app --host 127.0.0.1 --port 1235
```

#### Ruri/PyTorch Version (Legacy/Frozen)
```bash
./run_ruri_server.sh
```

### Automated Startup (macOS)
The project is configured to auto-start via `launchd` on macOS.
- **plist:** `~/Library/LaunchAgents/com.norihito.embed-reranker.plist`
- **Logs:** `~/Library/Logs/com.norihito.embed-reranker.log`

## Development Conventions

### Coding Style
- **Type Hinting:** Extensive use of Pydantic models for request/response validation.
- **Inference:** MLX implementation uses `mlx-embeddings` and `mlx-lm` for high-performance inference.
- **Portability:** Dual-backend support allows for development on various hardware while targeting MLX for production/local use on Mac.

### API Specifications
- **Health Check:** `GET /health` returns status and backend/model info.
- **Embedding:** `POST /v1/embeddings` (OpenAI-compatible format).
- **Rerank:** `POST /v1/rerank` or `POST /rerank` (Accepts `query`, `documents`, and `top_k`).

## Key Files
- `mlx_embed_rerank_server.py`: The high-performance MLX implementation.
- `ruri_embed_rerank_server.py`: The PyTorch-based implementation. (Legacy/Frozen)
- `run_mlx_server.sh`: MLX startup script using `uv`.
- `pyproject.toml`: Project metadata and dependencies (pinned to Python 3.13).
- `MIGRATION_SUMMARY.md`: Summary of the migration to `uv`.
- `MLX対応検討.md`: Design notes for MLX integration.
- `AUTO_STARTUP_SUMMARY.md`: Documentation for the `launchd` setup.
