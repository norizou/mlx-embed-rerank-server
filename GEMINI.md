# GEMINI.md - Project Context

## Project Overview
**embed-reranker** is a lightweight, Japanese-specialized API server that provides both Embedding and Reranking capabilities in a single process. It is optimized for RAG (Retrieval-Augmented Generation) workflows and is built entirely on the Apple Silicon native MLX backend for high performance.

### Key Technologies
- **Language:** Python 3.13+ (managed by `uv`)
- **Web Framework:** FastAPI + Uvicorn
- **AI/ML Backend:**
  - **MLX:** Native Apple Silicon optimization. The server runs completely on MLX.
- **Models:**
  - **Embedding:** `mlx-community/bge-m3-mlx-fp16` (Default), `mlx-community/embeddinggemma-300m-bf16`, `mlx-community/Qwen3-Embedding-0.6B-mxfp8`, `mlx-community/Qwen3-VL-Embedding-2B-mxfp8` (Multimodal)
  - **Reranker:** `mlx-community/Qwen3-Reranker-0.6B-mxfp8` (Default), `mlx-community/Qwen3-VL-Reranker-2B-mxfp8` (Multimodal)

### Architecture
The server exposes an OpenAI-compatible `/v1/embeddings` endpoint and a specialized `/v1/rerank` (or `/rerank`) endpoint. The MLX backend provides significantly lower latency and memory footprint on Mac hardware.

**Auto Fallback Feature:**
To optimize memory usage, the heavy Qwen3-VL models are automatically unloaded (paired unload) if there is no request for 30 seconds. The server then preloads the lightweight default models (`bge-m3` and `qwen3-0.6b`) and clears the Metal cache.

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

```bash
# Start via the MLX shell script
./run_mlx_server.sh

# Or directly via uv
uv run uvicorn mlx_embed_rerank_server:app --host 0.0.0.0 --port 1235
```

### Automated Startup (macOS)
The project is configured to auto-start via `launchd` on macOS.
- **plist:** `~/Library/LaunchAgents/com.norihito.embed-reranker.plist`
- **Logs:** `~/Library/Logs/com.norihito.embed-reranker.log`

## Development Conventions

### Coding Style
- **Type Hinting:** Extensive use of Pydantic models for request/response validation.
- **Inference:** MLX implementation uses `mlx-embeddings` and `mlx-lm` for high-performance inference.
- **Memory Management:** Implements thread-safe model caching and timeout-based memory freeing.

### API Specifications
- **Health Check:** `GET /health` returns status and loaded/available models.
- **Embedding:** `POST /v1/embeddings` (OpenAI-compatible format, supports `instruction` parameter for VLM).
- **Rerank:** `POST /v1/rerank` or `POST /rerank` (Accepts `query`, `documents`, `top_k`, and `instruction`).

## Key Files
- `mlx_embed_rerank_server.py`: The high-performance MLX implementation.
- `run_mlx_server.sh`: Startup and management script.
- `pyproject.toml`: Project metadata and dependencies (pinned to Python 3.13+).
- `README.md`: Comprehensive documentation including usage and auto-fallback sequences.
- `MIGRATION_SUMMARY.md`: Summary of the migration to `uv` and MLX/Qwen3-VL.
- `AUTO_STARTUP_SUMMARY.md`: Documentation for the `launchd` setup.
