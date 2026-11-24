# GEMINI.md - Project Context

## Project Overview
**embed-reranker** (package name: `mlx-embed-rerank-server`) is a lightweight, Japanese-focused API server that provides **Embedding**, **Reranking**, **STT (speech-to-text)** and **TTS (text-to-speech)** in a single process. It is optimized for RAG (Retrieval-Augmented Generation) workflows and is built entirely on the Apple Silicon native MLX backend.

### Key Technologies
- **Language:** Python 3.13+ (managed by `uv`)
- **Web Framework:** FastAPI + Uvicorn (API on port `1235`), plus a `http.server` health probe on port `1236` in a daemon thread
- **AI/ML Backend:** MLX (`mlx-embeddings`, `mlx-lm`, `mlx-audio`, `mlx-vlm` for EmbeddingGemma 2). No PyTorch path remains in the server.
- **Models:**
  - **Embedding (2 enabled):** `embeddinggemma-2-bf16` (default, text + image), `embeddinggemma-300m-bf16`. Disabled (commented out in `AVAILABLE_EMBED_MODELS`): `Qwen3-Embedding-0.6B-mxfp8` (2026-11-25; last on every task, BF16 no better — §12), `bge-m3`, `bge-m3-8bit` (HF cache removed 2026-11-24), `Qwen3-VL-Embedding-2B-mxfp8` (not cached, needs torch)
  - **Reranker (2):** `Qwen3-Reranker-0.6B-mxfp8` (default), `Qwen3-VL-Reranker-2B-mxfp8`
  - **ASR (1):** `Qwen3-ASR-1.7B-8bit` (default). `Qwen3-ASR-0.6B-8bit` disabled 2026-11-25 (only 0.25 s faster, much less accurate — §5)
  - **TTS — Qwen3 (2):** `Qwen3-TTS-12Hz-0.6B-Base-8bit`, `Qwen3-TTS-12Hz-1.7B-Base-8bit`
  - **TTS — Irodori (1):** `Irodori-TTS-v4.1-Small-8bit` (default) — Japanese-specialised flow-matching engine. One unified checkpoint covering voice cloning, VoiceDesign (caption) and automatic duration. The fp16 variant was removed (`BENCHMARK_REPORT.md` §7/§8 found it indistinguishable from 8bit on quality, worse on speed/memory/disk)

### Architecture
A single FastAPI app exposes OpenAI-compatible `/v1/embeddings`, `/v1/audio/transcriptions` and `/v1/audio/speech`, plus `/v1/rerank` (alias `/rerank`), `/v1/models` (OpenAI-style model list) and `/health`.

**Model Loading:**
All models are lazily loaded on the first request that needs them (nothing is preloaded at startup). `ModelManager` keeps four caches (`embed_cache`, `rerank_cache`, `asr_cache`, `tts_cache`) guarded by a single `threading.Lock`.

**Auto Fallback:**
A repeating 30-second timer unloads the Qwen3-VL embedding/reranker as a pair once either has been idle for 30 seconds (so 30–60 seconds in practice), calls `mx.metal.clear_cache()`, and preloads the default `embeddinggemma-2` / `qwen3-0.6b`. ASR/TTS models are never auto-unloaded.

**Health Probe:**
MLX inference blocks uvicorn's event loop, so a separate daemon thread serves `{"status": "ok"}` on port `1236`. `run_mlx_server.sh` monitors **1236** (not 1235) and restarts the process after 2 consecutive failures.

**Qwen3-VL requirement:**
`torch` / `torchvision` are **not** in `pyproject.toml`. Without them, `transformers`' `AutoImageProcessor` raises and any Qwen3-VL request returns HTTP 500. Install them explicitly (`uv pip install torch torchvision`) when those models are needed; every other model works without them — including `embeddinggemma-2` images, which use `Gemma4ImageProcessorPil` built from `processor_config.json` instead of `AutoProcessor` (whose video processor needs torch).

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
- **Embedding pooling is per model `type` — never one rule for all.** `standard` (BGE-M3) = CLS of `last_hidden_state` (mlx_embeddings' xlm_roberta `text_embeds` is *mean* pooling, so don't use it); `gemma` = `text_embeds` (mean → dense ×2 → L2); `qwen3_embed` = `text_embeds` (last token; the tokenizer appends `<|endoftext|>`); `embedding_gemma2` = `text_embeds` (mean) via mlx_vlm. Until 2026-11 every path took CLS, which left `gemma-3-300m` / `qwen3-0.6b-embed` near-useless (JMTEB-subset nDCG@10 3–18 vs 69–99) and made the old §2/§4 benchmark wrongly favour bge-m3. `TestRetrievalSanity` (4 queries × 4 docs) fails on the CLS path; keep it when adding models.
- **Default embedding:** `embeddinggemma-2` (768-d, text + image). bge-m3 is no longer registered, so a bge-m3 (1024-d) index must be re-embedded. Text-only retrieval is best on `gemma-3-300m` (`BENCHMARK_REPORT.md` §9).
- **Prompts:** `gemma` / `embedding_gemma2` prepend `task: search result | query: ` or `title: none | text: ` by `input_type` (images get none). `qwen3_embed` prepends `Instruct: {instruction}\nQuery:` to queries only. BGE-M3 ignores `input_type`.
- **`dimensions`:** Matryoshka truncation + re-normalisation, only for models with `"matryoshka": True`; 400 otherwise or when larger than the native dimension.
- **`instruction` parameter:** honored by `qwen3-0.6b-embed` (queries), `qwen3-vl-embedding-2b`, `qwen3-vl-reranker-2b`, and the default `qwen3-0.6b` reranker, where it fills `<Instruct>` (default `QWEN3_RERANK_DEFAULT_INSTRUCTION`, English as the model card recommends).
- **Rerank prompt:** `qwen3-0.6b` must use the model card's format (`QWEN3_RERANK_PREFIX` / `<Instruct>` / `<Query>` / `<Document>` / `QWEN3_RERANK_SUFFIX` with the empty `<think></think>` block), tokenised in one pass and scored one document per forward. The old prompt (no `<|im_end|>`, `<Instruct>` or `<think>`) lost 1.1–10.9 nDCG@10 and ranked below the embeddings alone on JaGovFAQs (BENCHMARK_REPORT.md §11); `TestRerankRetrievalSanity` guards against it. Documents are truncated to `RERANK_MAX_DOC_TOKENS` (1024).
- **Rerank scores:** P(yes) from a softmax over the `yes` / `no` logits, 0–1 (relevant ≈ 1, unrelated ≈ 0). `top_k` defaults to 10 and `0` means "return everything"; `top_n` (Cohere / Jina name) is an alias that wins when sent. OpenWebUI's external reranker sends only `top_n=len(documents)`, sorts the results by index and zips them with its documents by position, so cutting to fewer results than documents mis-assigns scores (`test_openwebui_external_reranker_gets_every_score`). Do not rerank an image+OCR document-page first stage: the reranker reads only text and lowers it (§11).
- **Errors:** unknown model names are `400` on every endpoint. `/v1/embeddings` validates model, images and `dimensions` before `get_embed()` (no multi-GB load for a bad request), re-raises `HTTPException`, and wraps only unexpected exceptions into a `500` whose `detail` carries the full traceback. Do not expose port 1235 publicly.
- **Irodori `caption` is not `ref_text`:** `caption` (alias `instruct`) is a *voice description* gated on `config.dit.use_caption_condition`, which v4.1 sets but older base variants do not. Never map the ICL transcript `ref_text` onto it. Irodori clones from `ref_audio` alone. The registry's `supports_caption` flag drives the `instruct` 400, so re-registering a caption-less variant restores the guard automatically.
- **Irodori duration:** a variant without a duration predictor falls back to `config.sampler.sequence_length` (750 frames = 30 s, ~24 GB) unless `seconds` is passed. `sequence_length` cannot be set through `generate()` — it is popped before the kwargs merge — so `seconds` is the only lever. v4.1 has a predictor, so this only matters if an older variant is re-registered.
- **Irodori multi-clip reference:** v4 accepts a list for `ref_audio` and encodes each clip separately before concatenating (120 s budget, overridable with `max_ref_seconds`). Qwen3-TTS takes a single path and returns 400 for a list.
- **Ignored vs rejected TTS params:** Qwen3-only params (`voice`, `ref_text`, `lang_code`, `max_tokens`) sent to Irodori are logged and ignored so OpenAI-compatible clients keep working; Irodori-only params sent to Qwen3, and `instruct` on an Irodori base variant, return `400`.
- **Irodori has no built-in default voice:** unlike Qwen3's `voice` presets, omitting both `ref_audio` and `instruct` on an Irodori model would silently zero out the reference embedding and generate an unconditioned, undefined voice instead of erroring. `/v1/audio/speech` rejects this with `400` before `get_tts()` runs. Since `DEFAULT_TTS` is now `irodori-tts-v4.1-small-8bit`, a bare call with only `input` (no `model`, no `ref_audio`/`instruct`) now 400s — Qwen3-TTS clients must pass `model` explicitly if they rely on the old model-omitted default.
- **Images:** only `/v1/embeddings` with a `supports_images` model (`embeddinggemma-2`) accepts them, as `{"type":"image_url","image_url":{"url":"data:..."}}` or `{"type":"image","image":"<base64>"}`, mixed with strings in any order. Only base64 is decoded — URLs and paths are 400 (SSRF / local-file read on a 0.0.0.0 server). One forward per image: `<bos><boi>{<image>×n}<eoi><eos>` with n = `num_soft_tokens_per_image`. Qwen3-VL models remain text-only.
- **`image_tokens`:** per-request image token budget, one of `IMAGE_TOKEN_OPTIONS` (70/140/280/560/1120; default `DEFAULT_IMAGE_TOKENS` = 280), passed to the PIL image processor as `max_soft_tokens` on each call — never set on the shared processor, so concurrent requests do not interfere. The processor rescales the image up or down to fill the budget. Validated before `get_embed()`: 400 on a model without `supports_images` or for any other value; ignored for text items. 560 is the sweet spot for text-heavy pages (BENCHMARK_REPORT.md §10: nDCG@10 69.7 → 75.2 on JDocQA; 1120 adds +0.5 at 2.6× the time).
- **Health:** `/health` returns `status`, `loaded_*_models` (4 lists) and `available_*` (3 lists). There is no `reranker_ready` field.

## Key Files
- `mlx_embed_rerank_server.py`: the entire server (models, ModelManager, endpoints, health thread).
- `run_mlx_server.sh`: startup, supervision (30s interval, 2 strikes, port 1236) and management.
- `pyproject.toml`: dependencies pinned to Python 3.13+.
- `README.md` / `README_JA.md`: full documentation (English / Japanese).
- `BENCHMARK_REPORT.md`: measured speed and accuracy for every model; the source of the default-model choices.
- `MIGRATION_SUMMARY.md`: uv and MLX migration history.
- `AUTO_STARTUP_SUMMARY.md`: launchd setup.
- `TTS_ENGINE_DESIGN.md`: why `/v1/audio/speech` carries two engines, how their parameters are split, and what the tests pin down. **Read this before touching the TTS endpoint.**
- `PLAN_STT_TTS_REVIEW.md`: pre-implementation design review for the audio endpoints (superseded in places by the shipped code).
- `tests/`: pytest API integration suite — `test_health.py`, `test_embeddings.py`, `test_rerank.py`, `test_audio.py`, `test_errors.py`, shared fixtures in `conftest.py`, and all expected values in `data/test_cases.json` (update it whenever `AVAILABLE_*_MODELS` changes). Design rationale: `tests/TEST_DESIGN.md`. To test a second server next to the live one: start it with `EMBED_HEALTH_PORT=1246 uv run uvicorn mlx_embed_rerank_server:app --port 1245` and run pytest with `EMBED_BASE_URL=http://127.0.0.1:1245 EMBED_HEALTH_URL=http://127.0.0.1:1246`. Markers: `vl` (auto-skipped without torch/torchvision), `audio` (slow; deselect with `-m "not audio"`).
