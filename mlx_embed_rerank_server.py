import math
import time
import io
import os
import uuid
import numpy as np
import mlx.core as mx
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any
import uvicorn
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json as _json

# MLX libraries
try:
    from mlx_embeddings import load as emb_load, generate as emb_generate
except ImportError:
    print("Warning: mlx-embeddings not found.")
    emb_load = None

try:
    from mlx_lm import load as lm_load
except ImportError:
    print("Warning: mlx-lm not found.")
    lm_load = None

# MLX Audio libraries
try:
    from mlx_audio.stt.utils import load_model as stt_load_model
except ImportError:
    print("Warning: mlx-audio STT not found.")
    stt_load_model = None

try:
    from mlx_audio.tts.utils import load_model as tts_load_model
except ImportError:
    print("Warning: mlx-audio TTS not found.")
    tts_load_model = None

try:
    from mlx_audio.audio_io import write as audio_write
except ImportError:
    print("Warning: mlx-audio audio_io not found.")
    audio_write = None

# =====================
# モデル設定定義
# =====================

AVAILABLE_EMBED_MODELS = {
    "gemma-3-300m": {
        "id": "mlx-community/embeddinggemma-300m-bf16",
        "type": "gemma",
        "description": "Gemma 3 300M (Multilingual, Prefixes required)"
    },
    "bge-m3": {
        "id": "mlx-community/bge-m3-mlx-fp16",
        "type": "standard",
        "description": "BGE-M3 (Multilingual, High precision)"
    },
    "bge-m3-8bit": {
        "id": "mlx-community/bge-m3-mlx-8bit",
        "type": "standard",
        "description": "BGE-M3 8-bit (Multilingual, Quantized)"
    },
    "qwen3-vl-embedding-2b": {
        "id": "mlx-community/Qwen3-VL-Embedding-2B-mxfp8",
        "type": "qwen3_vl_embed",
        "description": "Qwen3-VL Embedding 2B (Multimodal, mlx-embeddings)"
    },
    "qwen3-0.6b-embed": {
        "id": "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
        "type": "standard",
        "description": "Qwen3 Embedding 0.6B (Text only)"
    }
}

AVAILABLE_RERANK_MODELS = {
    "qwen3-0.6b": {
        "id": "mlx-community/Qwen3-Reranker-0.6B-mxfp8",
        "type": "qwen3",
        "description": "Qwen3 Reranker 0.6B (Generative, Yes/No scoring)"
    },
    "qwen3-vl-reranker-2b": {
        "id": "mlx-community/Qwen3-VL-Reranker-2B-mxfp8",
        "type": "qwen3_vl_rerank",
        "description": "Qwen3-VL Reranker 2B (Multimodal, mlx-embeddings)"
    }
}

AVAILABLE_AUDIO_MODELS = {
    "qwen3-asr-0.6b-8bit": {
        "id": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "type": "asr",
        "description": "Qwen3 ASR 0.6B 8-bit (Speech-to-Text, Fast)"
    },
    "qwen3-asr-1.7b-8bit": {
        "id": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "type": "asr",
        "description": "Qwen3 ASR 1.7B 8-bit (Speech-to-Text, Best accuracy)"
    },
    "qwen3-tts-0.6b-base-8bit": {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "type": "tts",
        "description": "Qwen3 TTS 0.6B Base 8-bit (Text-to-Speech, Voice Clone)"
    },
    "qwen3-tts-1.7b-base-8bit": {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "type": "tts",
        "description": "Qwen3 TTS 1.7B Base 8-bit (Text-to-Speech, Voice Clone, Stable)"
    },
    # --- Irodori TTS (日本語特化, Flow Matching) ---
    # v4.1-Small は単一チェックポイントでボイスクローン / VoiceDesign (caption) /
    # 出力長の自動推定をすべて備える。ModernBERT-ja テキストエンコーダと DACVAE を
    # 同梱するため、推論時の追加ダウンロードが発生しない。
    "irodori-tts-v4.1-small-8bit": {
        "id": "mlx-community/Irodori-TTS-v4.1-Small-8bit",
        "type": "tts_irodori",
        "supports_caption": True,
        "description": "Irodori TTS v4.1 Small 8-bit (Japanese TTS, Voice Clone + VoiceDesign + Auto Duration)"
    },
}

# /v1/audio/speech が受け付けるモデル種別。エンジンごとに generate() の
# キーワード引数が異なるため、種別名でディスパッチする。
TTS_TYPES = {"tts", "tts_irodori"}

DEFAULT_EMBED = "bge-m3"
DEFAULT_RERANK = "qwen3-0.6b"
DEFAULT_ASR = "qwen3-asr-1.7b-8bit"
DEFAULT_TTS = "irodori-tts-v4.1-small-8bit"

# =====================
# モデルマネージャ
# =====================

class ModelManager:
    def __init__(self, inactivity_timeout: int = 30):
        self.embed_cache = {}
        self.rerank_cache = {}
        self.asr_cache = {}
        self.tts_cache = {}
        self.last_used = {}
        self.lock = threading.Lock()
        self.inactivity_timeout = inactivity_timeout
        self._start_fallback_timer()

    def _start_fallback_timer(self):
        self._fallback_timer = threading.Timer(self.inactivity_timeout, self._check_inactivity)
        self._fallback_timer.daemon = True
        self._fallback_timer.start()

    def _check_inactivity(self):
        now = time.time()
        with self.lock:
            qwen3vl_embed_timedout = False
            qwen3vl_rerank_timedout = False

            for name, ts in list(self.last_used.items()):
                if name in self.embed_cache:
                    config = AVAILABLE_EMBED_MODELS.get(name, {})
                    if config.get("type", "").startswith("qwen3_vl") and now - ts > self.inactivity_timeout:
                        qwen3vl_embed_timedout = True

            for name, ts in list(self.last_used.items()):
                if name in self.rerank_cache:
                    config = AVAILABLE_RERANK_MODELS.get(name, {})
                    if config.get("type", "").startswith("qwen3_vl") and now - ts > self.inactivity_timeout:
                        qwen3vl_rerank_timedout = True

            if qwen3vl_embed_timedout or qwen3vl_rerank_timedout:
                # qwen3-vl のどちらかが timeout したら両方 unload
                for name in list(self.embed_cache.keys()):
                    config = AVAILABLE_EMBED_MODELS.get(name, {})
                    if config.get("type", "").startswith("qwen3_vl"):
                        print(f"[Fallback] Unload embed '{name}' (paired unload)")
                        self.embed_cache.pop(name, None)
                        self.last_used.pop(name, None)
                for name in list(self.rerank_cache.keys()):
                    config = AVAILABLE_RERANK_MODELS.get(name, {})
                    if config.get("type", "").startswith("qwen3_vl"):
                        print(f"[Fallback] Unload rerank '{name}' (paired unload)")
                        self.rerank_cache.pop(name, None)
                        self.last_used.pop(name, None)

                # デフォルトモデルをプリロード
                if DEFAULT_EMBED not in self.embed_cache:
                    print(f"[Fallback] Preload default embed '{DEFAULT_EMBED}'")
                    try:
                        self._load_embed_unlocked(DEFAULT_EMBED, now)
                    except Exception as e:
                        print(f"[Fallback] Failed to preload default embed: {e}")
                if DEFAULT_RERANK not in self.rerank_cache:
                    print(f"[Fallback] Preload default rerank '{DEFAULT_RERANK}'")
                    try:
                        self._load_rerank_unlocked(DEFAULT_RERANK, now)
                    except Exception as e:
                        print(f"[Fallback] Failed to preload default rerank: {e}")

                try:
                    mx.metal.clear_cache()
                except Exception:
                    pass

        self._start_fallback_timer()

    def _fix_qwen3vl_processor(self, model_processor_tuple, repo_id: str):
        """Work around mlx_embeddings not calling Qwen3VLProcessor.__init__."""
        _model, processor = model_processor_tuple
        inner = getattr(processor, "processor", None)
        if inner is not None and inner.__class__.__name__ == "Qwen3VLProcessor":
            try:
                from transformers import AutoProcessor
                correct = AutoProcessor.from_pretrained(repo_id)
                for attr in ["image_ids", "video_ids", "audio_ids", "chat_template"]:
                    if hasattr(correct, attr):
                        setattr(inner, attr, getattr(correct, attr))
            except Exception:
                pass
        return model_processor_tuple

    def _load_embed_unlocked(self, name: str, now: float):
        if name in self.embed_cache:
            return
        config = AVAILABLE_EMBED_MODELS[name]
        print(f"Loading Embedding model: {config['id']}...")
        loaded = emb_load(config['id'])
        if config.get("type", "").startswith("qwen3_vl"):
            loaded = self._fix_qwen3vl_processor(loaded, config['id'])
        self.embed_cache[name] = loaded
        self.last_used[name] = now

    def get_embed(self, name: str):
        if name not in AVAILABLE_EMBED_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported embedding model: {name}")
        with self.lock:
            if name not in self.embed_cache:
                self._load_embed_unlocked(name, time.time())
            self.last_used[name] = time.time()
            return self.embed_cache[name], AVAILABLE_EMBED_MODELS[name]

    def _load_rerank_unlocked(self, name: str, now: float):
        if name in self.rerank_cache:
            return
        config = AVAILABLE_RERANK_MODELS[name]
        print(f"Loading Reranker model: {config['id']}...")
        if config["type"] == "qwen3_vl_rerank":
            loaded = emb_load(config['id'])
            loaded = self._fix_qwen3vl_processor(loaded, config['id'])
            self.rerank_cache[name] = loaded
        else:
            model, tokenizer = lm_load(config['id'])
            yes_token_id = None
            no_token_id = None
            if config["type"] == "qwen3":
                yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[-1]
                no_token_id = tokenizer.encode("no", add_special_tokens=False)[-1]
            self.rerank_cache[name] = {
                "model": model,
                "tokenizer": tokenizer,
                "yes_token_id": yes_token_id,
                "no_token_id": no_token_id
            }
        self.last_used[name] = now

    def get_rerank(self, name: str):
        if name not in AVAILABLE_RERANK_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported rerank model: {name}")
        with self.lock:
            if name not in self.rerank_cache:
                self._load_rerank_unlocked(name, time.time())
            self.last_used[name] = time.time()
            return self.rerank_cache[name], AVAILABLE_RERANK_MODELS[name]

    def get_asr(self, name: str):
        if name not in AVAILABLE_AUDIO_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported ASR model: {name}")
        config = AVAILABLE_AUDIO_MODELS[name]
        if config["type"] != "asr":
            raise HTTPException(status_code=400, detail=f"Model {name} is not an ASR model")
        with self.lock:
            if name not in self.asr_cache:
                print(f"Loading ASR model: {config['id']}...")
                if stt_load_model is None:
                    raise HTTPException(status_code=500, detail="mlx-audio STT not available")
                self.asr_cache[name] = stt_load_model(config['id'])
                self.last_used[name] = time.time()
            self.last_used[name] = time.time()
            return self.asr_cache[name]

    def get_tts(self, name: str):
        if name not in AVAILABLE_AUDIO_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported TTS model: {name}")
        config = AVAILABLE_AUDIO_MODELS[name]
        if config["type"] not in TTS_TYPES:
            raise HTTPException(status_code=400, detail=f"Model {name} is not a TTS model")
        with self.lock:
            if name not in self.tts_cache:
                print(f"Loading TTS model: {config['id']}...")
                if tts_load_model is None:
                    raise HTTPException(status_code=500, detail="mlx-audio TTS not available")
                self.tts_cache[name] = tts_load_model(config['id'])
                self.last_used[name] = time.time()
            self.last_used[name] = time.time()
            return self.tts_cache[name]

manager = ModelManager()

# =====================
# 推論ロジック
# =====================

def compute_embeddings(texts: List[str], model_name: str, input_type: str = "document", instruction: Optional[str] = None) -> List[List[float]]:
    (model, processor), config = manager.get_embed(model_name)
    
    # プレフィックス処理
    if config["type"] == "gemma":
        prefix = "task: search result | query: " if input_type == "query" else "title: none | text: "
        texts = [f"{prefix}{t}" for t in texts]
    
    # Qwen3-VL Embedding モデル
    if config["type"] == "qwen3_vl_embed":
        inst = instruction or (
            "Represent this sentence for retrieving relevant documents."
            if input_type == "query"
            else "Represent this document for retrieval."
        )
        inputs = [{"text": t, "instruction": inst} for t in texts]
        embeddings = model.process(inputs, processor=processor)
        arr = np.array(embeddings)
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1e-10, norm)
        return (arr / norm).tolist()
    
    # gemmaモデルは __call__ の引数名が 'inputs' (mlx_embeddings.generate は 'input_ids' を渡すためエラー)
    if config["type"] == "gemma":
        tokenized = processor(texts, return_tensors="np", padding=True, truncation=True)
        input_data = {k: mx.array(v) for k, v in tokenized.items()}
        # input_ids → inputs にリネーム
        if "input_ids" in input_data and "inputs" not in input_data:
            input_data["inputs"] = input_data.pop("input_ids")
        output = model(**input_data)
    else:
        output = emb_generate(model, processor, texts=texts)

    if hasattr(output, "last_hidden_state"):
        state = np.array(output.last_hidden_state)
        cls_embeddings = state[:, 0, :]
        norm = np.linalg.norm(cls_embeddings, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1e-10, norm)
        return (cls_embeddings / norm).tolist()
    elif hasattr(output, "pooler_output"):
        return np.array(output.pooler_output).tolist()
    elif hasattr(output, "embeddings"):
        return np.array(output.embeddings).tolist()
    if hasattr(output, "tolist"):
        return output.tolist()

    raise HTTPException(status_code=500, detail="Failed to process embedding output")

def compute_rerank(query: str, docs: List[str], model_name: str, instruction: Optional[str] = None) -> List[float]:
    cached, config = manager.get_rerank(model_name)
    
    # Qwen3-VL Reranker
    if config["type"] == "qwen3_vl_rerank":
        model, processor = cached
        inst = instruction or (
            "Judge whether the Document meets the requirements based on the Query. "
            "Note that the answer can only be \"yes\" or \"no\"."
        )
        inputs = {
            "instruction": inst,
            "query": {"text": query},
            "documents": [{"text": d} for d in docs],
        }
        scores = model.process(inputs, processor=processor)
        mx.eval(scores)
        return [float(s) for s in scores.tolist()]
    
    model = cached["model"]
    tokenizer = cached["tokenizer"]

    scores = []
    for doc in docs:
        if config["type"] == "qwen3":
            prompt = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. "
                "Note that the answer can only be \"yes\" or \"no\".\n"
                "<|im_start|>user\n"
                f"<Query>: {query}\n<Document>: {doc}\n"
                "<|im_start|>assistant\n"
            )
            input_ids = mx.array(tokenizer.encode(prompt))
            logits = model(input_ids[None, :])
            last_logits = logits[0, -1, :]
            
            yes_l = float(last_logits[cached["yes_token_id"]])
            no_l = float(last_logits[cached["no_token_id"]])
            m = max(yes_l, no_l)
            scores.append(math.exp(yes_l - m) / (math.exp(yes_l - m) + math.exp(no_l - m)))
        else:
            # 標準的な CrossEncoder (BGE等) は通常 [CLS] トークンのロジットを使用
            # ここでは簡易的な実装として mlx-lm の出力を利用
            prompt = f"Query: {query} Document: {doc} Relevant:"
            input_ids = mx.array(tokenizer.encode(prompt))
            logits = model(input_ids[None, :])
            # 最終トークンの最大ロジットをスコアとして使用（モデルにより調整が必要）
            score = float(mx.max(logits[0, -1, :]))
            scores.append(score)
            
    return scores
# =====================
# API定義
# =====================

app = FastAPI(title="MLX Multi-Model Embedding + Rerank Server")

class EmbReq(BaseModel):
    model: Optional[str] = DEFAULT_EMBED
    input: Union[str, List[str]]
    input_type: Optional[str] = "document"
    instruction: Optional[str] = None

@app.post("/v1/embeddings")
async def embeddings(req: EmbReq):
    import traceback
    try:
        texts = [req.input] if isinstance(req.input, str) else req.input
        model_name = req.model or DEFAULT_EMBED
        vecs = compute_embeddings(texts, model_name, req.input_type, req.instruction)
        
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
            "model": model_name,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Embedding error: {tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{tb}")

class RerankReq(BaseModel):
    model: Optional[str] = DEFAULT_RERANK
    query: str
    documents: List[str]
    top_k: int = 10
    instruction: Optional[str] = None

@app.post("/v1/rerank")
@app.post("/rerank")
async def rerank(req: RerankReq):
    model_name = req.model or DEFAULT_RERANK
    scores = compute_rerank(req.query, req.documents, model_name, req.instruction)

    ranked = sorted(
        ({"index": i, "relevance_score": scores[i], "document": req.documents[i]} for i in range(len(req.documents))),
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    if req.top_k:
        ranked = ranked[: min(req.top_k, len(ranked))]

    for item in ranked:
        item.pop("document")

    return {
        "model": model_name,
        "results": ranked,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_embed_models": list(manager.embed_cache.keys()),
        "loaded_rerank_models": list(manager.rerank_cache.keys()),
        "loaded_asr_models": list(manager.asr_cache.keys()),
        "loaded_tts_models": list(manager.tts_cache.keys()),
        "available_embed": list(AVAILABLE_EMBED_MODELS.keys()),
        "available_rerank": list(AVAILABLE_RERANK_MODELS.keys()),
        "available_audio": list(AVAILABLE_AUDIO_MODELS.keys()),
    }

# =====================
# Audio Endpoints
# =====================

class TranscriptionReq(BaseModel):
    model: Optional[str] = DEFAULT_ASR
    language: Optional[str] = None

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_ASR),
    language: Optional[str] = Form(None),
):
    """Transcribe audio using an STT model (OpenAI-compatible)."""
    if audio_write is None:
        raise HTTPException(status_code=500, detail="mlx-audio audio_io not available")

    # 1. アップロードされた音声を一時ファイルに書き出し
    #    (mlx-audio の audio_io.read(BytesIO) は m4a 等の moov 末尾配置で
    #     ffmpeg へのパイプ入力がシーク不可となり空配列を返すバグがあるため、
    #     bytes を直接ファイルに書いてモデル内部の load_audio にファイルパスを渡す)
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = f"/tmp/stt_{uuid.uuid4()}{suffix}"
    data = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(data)

    try:
        # 2. モデルをlazy load
        stt_model = manager.get_asr(model)

        # 3. 推論実行
        kwargs = {}
        if language:
            kwargs["language"] = language
        result = stt_model.generate(tmp_path, **kwargs)

        # 4. resultから.textを取得して返却
        text = result.text if hasattr(result, "text") else str(result)
        return {"text": text}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

class SpeechReq(BaseModel):
    input: str
    model: Optional[str] = DEFAULT_TTS
    response_format: Optional[str] = "mp3"
    # --- 共通 ---
    speed: Optional[float] = 1.0
    # Irodori v4 系はクリップのリストを受け取り、各クリップを個別にエンコードして
    # 連結する (学習時の形式に一致)。Qwen3-TTS は単一パスのみ。
    ref_audio: Optional[Union[str, List[str]]] = None
    # --- Qwen3-TTS 専用 ---
    voice: Optional[str] = None
    ref_text: Optional[str] = None
    lang_code: Optional[str] = "auto"
    max_tokens: Optional[int] = None
    # --- Irodori 専用 ---
    instruct: Optional[str] = None          # 声質の記述 (caption 条件を持つモデルのみ)
    seconds: Optional[float] = None         # 出力長を秒で明示
    duration_scale: Optional[float] = None  # 推定された長さに対する倍率
    num_steps: Optional[int] = None         # Euler ステップ数
    cfg_guidance_mode: Optional[str] = None # independent / alternating
    max_ref_seconds: Optional[float] = None # 参照音声の上限秒 (既定はモデルの 120s)
    # duration predictor の推定値をクランプする範囲。既定は 0.5〜30 秒で、
    # 長文では上限に張り付いて頭打ちになるため明示的に引き上げられるようにする。
    min_seconds: Optional[float] = None
    max_seconds: Optional[float] = None

@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechReq):
    """Generate speech audio from text (OpenAI-compatible)."""
    if audio_write is None:
        raise HTTPException(status_code=500, detail="mlx-audio audio_io not available")

    model_name = req.model or DEFAULT_TTS

    # 数 GB のモデルロードを起こす前にリクエストを検証する。
    # モデル種別のエラーは get_tts() と同じ契約を保つ。
    if model_name not in AVAILABLE_AUDIO_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported TTS model: {model_name}")
    config = AVAILABLE_AUDIO_MODELS[model_name]
    tts_engine = config["type"]
    if tts_engine not in TTS_TYPES:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not a TTS model")

    irodori_only = {
        "instruct": req.instruct,
        "seconds": req.seconds,
        "duration_scale": req.duration_scale,
        "num_steps": req.num_steps,
        "cfg_guidance_mode": req.cfg_guidance_mode,
        "max_ref_seconds": req.max_ref_seconds,
        "min_seconds": req.min_seconds,
        "max_seconds": req.max_seconds,
    }
    if tts_engine == "tts_irodori":
        # caption 条件を持たない Irodori 版 (v2/v3 base 等) を登録した場合、
        # instruct は黙って捨てられるので拒否する。v4.1 は caption を持つ。
        if req.instruct and not config.get("supports_caption"):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} has no caption conditioning and cannot use 'instruct'.",
            )
        # Irodori には Qwen3 の voice プリセットに相当する既定話者が無い。両方省略すると
        # 参照埋め込みがゼロ化された無条件生成になり、エラーにならず素性不明の声が返る
        # ため、黙って通さず拒否する。
        if not req.ref_audio and not req.instruct:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} has no built-in default voice; "
                       f"pass 'ref_audio' (voice clone) or 'instruct' (VoiceDesign).",
            )
    else:
        if isinstance(req.ref_audio, list):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} accepts a single ref_audio path, not a list. "
                       f"Multi-clip references are an Irodori v4 feature.",
            )
        supplied = [k for k, v in irodori_only.items() if v is not None]
        if supplied:
            raise HTTPException(
                status_code=400,
                detail=f"Parameters {supplied} are Irodori-only and not supported by model {model_name}.",
            )

    tts_model = manager.get_tts(model_name)

    # model.generate() はジェネレータ
    generate_kwargs = {}

    if tts_engine == "tts_irodori":
        # Irodori はリファレンス音声だけでクローンする (書き起こしは不要)。
        # caption は声質を言葉で指示する VoiceDesign 用の条件で、ref_text とは
        # 別物なのでマッピングしない。v4.1 は ref_audio と caption を併用できる。
        if req.ref_audio:
            generate_kwargs["ref_audio"] = req.ref_audio
        if req.instruct:
            generate_kwargs["caption"] = req.instruct
        if req.seconds is not None:
            generate_kwargs["seconds"] = req.seconds
        # speed は「大きいほど速い」、duration_scale は「大きいほど長い」で逆向き
        if req.duration_scale is not None:
            generate_kwargs["duration_scale"] = req.duration_scale
        elif req.speed and req.speed != 1.0:
            generate_kwargs["duration_scale"] = 1.0 / req.speed
        if req.num_steps:
            generate_kwargs["num_steps"] = req.num_steps
        if req.cfg_guidance_mode:
            generate_kwargs["cfg_guidance_mode"] = req.cfg_guidance_mode
        if req.max_ref_seconds is not None:
            generate_kwargs["max_ref_seconds"] = req.max_ref_seconds
        if req.min_seconds is not None:
            generate_kwargs["min_seconds"] = req.min_seconds
        if req.max_seconds is not None:
            generate_kwargs["max_seconds"] = req.max_seconds

        ignored = [k for k in ("voice", "ref_text", "max_tokens") if getattr(req, k)]
        if req.lang_code and req.lang_code != "auto":
            ignored.append("lang_code")
        if ignored:
            print(f"Warning: {ignored} are Qwen3-TTS parameters and are ignored by {model_name}.")

        # duration predictor 非搭載のモデル (v2 等) を登録した場合、seconds 未指定だと
        # 30 秒固定 (sequence_length=750) になり約 24GB を要する。v4.1 は搭載済み。
        dit_cfg = getattr(getattr(tts_model, "config", None), "dit", None)
        if req.seconds is None and not getattr(dit_cfg, "use_duration_predictor", True):
            print(f"Warning: {model_name} has no duration predictor; generating a fixed 30s "
                  f"(sequence_length=750, ~24GB). Pass 'seconds' to cut memory and time.")
    else:
        if req.voice:
            generate_kwargs["voice"] = req.voice
        if req.speed and req.speed != 1.0:
            generate_kwargs["speed"] = req.speed
        if req.ref_audio:
            generate_kwargs["ref_audio"] = req.ref_audio
        if req.ref_text:
            generate_kwargs["ref_text"] = req.ref_text
        if req.lang_code and req.lang_code != "auto":
            generate_kwargs["lang_code"] = req.lang_code
        # ICL (ref_audio + ref_text のボイスクローン) は split_pattern による分割を行わず
        # 入力テキスト全体を 1 回の生成で処理するため、既定の max_tokens=4096 では
        # 長文が途中で切れる。非 ICL 経路は max_tokens が「セグメントあたり」なので既定で足りる。
        if req.max_tokens:
            generate_kwargs["max_tokens"] = req.max_tokens
        elif req.ref_audio and req.ref_text:
            generate_kwargs["max_tokens"] = 8192

    # 全チャンクを収集
    audio_chunks = []
    sample_rate = None
    for result in tts_model.generate(req.input, **generate_kwargs):
        audio_chunks.append(result.audio)
        if sample_rate is None:
            sample_rate = result.sample_rate

    if not audio_chunks:
        raise HTTPException(status_code=400, detail="No audio generated")

    # 結合してエンコード
    concatenated = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    audio_write(buffer, concatenated, sample_rate, format=req.response_format)
    buffer.seek(0)

    content_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }

    return StreamingResponse(
        buffer,
        media_type=content_type_map.get(req.response_format, "audio/mpeg"),
        headers={
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}"
        },
    )

# =====================
# ヘルスチェック専用サーバー (別スレッド)
# MLX推論でメインのイベントループがブロックされても
# スーパーバイザーのヘルスチェックに応答できるようにする
# =====================
HEALTH_PORT = 1236

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = _json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress access logs

def _start_health_server():
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), _HealthHandler)
    server.serve_forever()

_health_thread = threading.Thread(target=_start_health_server, daemon=True)
_health_thread.start()
print(f"Health-check server started on port {HEALTH_PORT}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1235)
