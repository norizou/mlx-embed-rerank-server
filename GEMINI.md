# GEMINI.md - Project Context

## Project Overview
**embed-reranker** (package name: `mlx-embed-rerank-server`) is a lightweight, Japanese-focused API server that provides **Embedding**, **Reranking**, **STT (speech-to-text)** and **TTS (text-to-speech)** in a single process. It is optimized for RAG (Retrieval-Augmented Generation) workflows and is built entirely on the Apple Silicon native MLX backend.

### Key Technologies
- **Language:** Python 3.13+ (managed by `uv`)
- **Web Framework:** FastAPI + Uvicorn (API on port `1235`), plus a `http.server` health probe on port `1236` in a daemon thread
- **AI/ML Backend:** MLX (`mlx-embeddings`, `mlx-lm`, `mlx-audio`). No PyTorch path remains in the server.
- **Models:**
  - **Embedding (5):** `bge-m3` (default), `bge-m3-8bit`, `embeddinggemma-300m-bf16`, `Qwen3-Embedding-0.6B-mxfp8`, `Qwen3-VL-Embedding-2B-mxfp8`
  - **Reranker (2):** `Qwen3-Reranker-0.6B-mxfp8` (default), `Qwen3-VL-Reranker-2B-mxfp8`
  - **ASR (2):** `Qwen3-ASR-1.7B-8bit` (default), `Qwen3-ASR-0.6B-8bit`
  - **TTS — Qwen3 (2):** `Qwen3-TTS-12Hz-0.6B-Base-8bit` (default), `Qwen3-TTS-12Hz-1.7B-Base-8bit`
  - **TTS — Irodori (6):** `Irodori-TTS-500M-v3-{fp16,8bit}`, `Irodori-TTS-500M-v2-{fp16,8bit}`, `Irodori-TTS-600M-v3-VoiceDesign-{fp16,8bit}` — Japanese-specialised flow-matching engine, registered provisionally pending a benchmark

### Architecture
A single FastAPI app exposes OpenAI-compatible `/v1/embeddings`, `/v1/audio/transcriptions` and `/v1/audio/speech`, plus `/v1/rerank` (alias `/rerank`) and `/health`.

**Model Loading:**
All models are lazily loaded on the first request that needs them (nothing is preloaded at startup). `ModelManager` keeps four caches (`embed_cache`, `rerank_cache`, `asr_cache`, `tts_cache`) guarded by a single `threading.Lock`.

**Auto Fallback:**
A repeating 30-second timer unloads the Qwen3-VL embedding/reranker as a pair once either has been idle for 30 seconds (so 30–60 seconds in practice), calls `mx.metal.clear_cache()`, and preloads the default `bge-m3` / `qwen3-0.6b`. ASR/TTS models are never auto-unloaded.

**Health Probe:**
MLX inference blocks uvicorn's event loop, so a separate daemon thread serves `{"status": "ok"}` on port `1236`. `run_mlx_server.sh` monitors **1236** (not 1235) and restarts the process after 2 consecutive failures.

**Qwen3-VL requirement:**
`torch` / `torchvision` are **not** in `pyproject.toml`. Without them, `transformers`' `AutoImageProcessor` raises and any Qwen3-VL request returns HTTP 500. Install them explicitly (`uv pip install torch torchvision`) when those models are needed; every other model works without them.

## Building and Running

### Prerequisites
- Python 3.13 or higher, `uv`, Apple Silicon macOS
- `ffmpeg` for mp4/m4a/webm audio input (`brew install ffmpeg`)

### Development Setup
```bash
uv sync           # runtime deps
uv sync --extra dev   # + pytest / httpx
```

### Running the Server
```bash
# Start via the supervisor script (recommended)
./run_mlx_server.sh
./run_mlx_server.sh status | restart | kill

# Or directly
uv run uvicorn mlx_embed_rerank_server:app --host 0.0.0.0 --port 1235
```

### Automated Startup (macOS)
- **plist:** `~/Library/LaunchAgents/com.norihito.embed-reranker.plist` (RunAtLoad + KeepAlive)
- **Script path in plist:** `/Users/norihito/Projects/AI/Workspace/embed_reranker/run_mlx_server.sh`
- **Logs:** `~/Library/Logs/com.norihito.embed-reranker.log` / `.error.log`

## Development Conventions

### Coding Style
- **Type Hinting:** Pydantic models (`EmbReq`, `RerankReq`, `SpeechReq`) for JSON request validation; the STT endpoint uses `File`/`Form` parameters instead.
- **Inference:** `mlx-embeddings` for embedding/VL models, `mlx-lm` for the generative reranker, `mlx-audio` for ASR/TTS.
- **TTS engine dispatch:** `AVAILABLE_AUDIO_MODELS[...]["type"]` selects the engine (`tts` = Qwen3, `tts_irodori` = Irodori); `TTS_TYPES` is the set `/v1/audio/speech` accepts. The two engines take disjoint `generate()` kwargs, so `/v1/audio/speech` branches on the type and validates the request **before** `get_tts()` runs (an invalid request must never trigger a multi-GB load).
- **Memory Management:** thread-safe model caching plus timeout-based unloading for Qwen3-VL only.

### API Behaviour Notes (keep in sync when editing)
- **Embedding output:** every model path takes the first token of `last_hidden_state` (CLS) and L2-normalizes it. This matches BGE-M3's dense-embedding spec; `bge-m3` is therefore the recommended default (see `BENCHMARK_REPORT.md` §4, where `gemma-3-300m` / `qwen3-0.6b-embed` missed the target document in top-100 while `bge-m3` reached recall 100%).
- **`instruction` parameter:** honored only by `qwen3-vl-embedding-2b` and `qwen3-vl-reranker-2b`. The default `qwen3-0.6b` reranker uses a fixed yes/no prompt and ignores it.
- **Rerank scores:** softmax over the `yes` / `no` logits, so absolute values are small; the ordering is what matters. `top_k` defaults to 10 and `0` means "return everything".
- **Errors:** `/v1/rerank` and the audio endpoints return `400` for unknown model names; `/v1/embeddings` wraps every exception (including that 400) into a `500` whose `detail` carries the full traceback. Do not expose port 1235 publicly.
- **Irodori `caption` is not `ref_text`:** `caption` (alias `instruct`) is a *voice description* used only by the VoiceDesign variants — `config.dit.use_caption_condition` gates it, so base variants drop it silently. Never map the ICL transcript `ref_text` onto it. Irodori clones from `ref_audio` alone.
- **Irodori v2 duration:** v2 has no duration predictor, so without `seconds` it falls back to `config.sampler.sequence_length` (750 frames = 30 s, ~24 GB). `sequence_length` cannot be set through `generate()` — it is popped before the kwargs merge — so `seconds` is the only lever.
- **Ignored vs rejected TTS params:** Qwen3-only params (`voice`, `ref_text`, `lang_code`, `max_tokens`) sent to Irodori are logged and ignored so OpenAI-compatible clients keep working; Irodori-only params sent to Qwen3, and `instruct` on an Irodori base variant, return `400`.
- **Images:** the HTTP API is text-only. Qwen3-VL models are served as text embedders/rerankers.
- **Health:** `/health` returns `status`, `loaded_*_models` (4 lists) and `available_*` (3 lists). There is no `reranker_ready` field.

## Key Files
- `mlx_embed_rerank_server.py`: the entire server (models, ModelManager, endpoints, health thread).
- `run_mlx_server.sh`: startup, supervision (30s interval, 2 strikes, port 1236) and management.
- `pyproject.toml`: dependencies pinned to Python 3.13+.
- `README.md` / `README_JA.md`: full documentation (English / Japanese).
- `BENCHMARK_REPORT.md`: measured speed and accuracy for every model; the source of the default-model choices.
- `MIGRATION_SUMMARY.md`: uv and MLX migration history.
- `AUTO_STARTUP_SUMMARY.md`: launchd setup.
- `PLAN_STT_TTS_REVIEW.md`: pre-implementation design review for the audio endpoints (superseded in places by the shipped code).
- `tests/`: pytest API integration suite — `test_health.py`, `test_embeddings.py`, `test_rerank.py`, `test_audio.py`, `test_errors.py`, shared fixtures in `conftest.py`, and all expected values in `data/test_cases.json` (update it whenever `AVAILABLE_*_MODELS` changes). Design rationale: `tests/TEST_DESIGN.md`. Markers: `vl` (auto-skipped without torch/torchvision), `audio` (slow; deselect with `-m "not audio"`).
