import math
import time
import io
import os

# Disable Hugging Face tqdm progress bars to prevent BrokenPipeError in background/daemon execution
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uuid
import datetime
import _strptime
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
    from mlx_vlm.utils import get_model_path as vlm_get_model_path, load_model as vlm_load_model
except ImportError:
    print("Warning: mlx-vlm not found (embeddinggemma-2 unavailable).")
    vlm_get_model_path = vlm_load_model = None

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

# "type" はプーリング方式と前処理を決める。各モデルの公式仕様に合わせること:
#   standard         : CLS (先頭トークン) + L2。BGE-M3 の dense 埋め込みの仕様。
#                      (mlx_embeddings の xlm_roberta.text_embeds は mean pooling なので使わない)
#   gemma            : EmbeddingGemma。mean pooling -> dense x2 -> L2 (= mlx_embeddings の text_embeds)
#   qwen3_embed      : Qwen3-Embedding。末尾 <|endoftext|> の last-token pooling -> L2 (= text_embeds)。
#                      クエリには指示文を付ける
#   embedding_gemma2 : EmbeddingGemma 2。mlx_vlm でロード、mean pooling -> L2 (= text_embeds)。画像も可
#   qwen3_vl_embed   : Qwen3-VL Embedding。model.process() に任せる
# "matryoshka": True のモデルは `dimensions` で先頭 N 次元に切り詰めて再正規化できる。
# "supports_images": True のモデルは画像を受け付け、`image_tokens` で 1 枚あたりの画像トークン数を選べる。
AVAILABLE_EMBED_MODELS = {
    "gemma-3-300m": {
        "id": "mlx-community/embeddinggemma-300m-bf16",
        "type": "gemma",
        "matryoshka": True,
        "description": "EmbeddingGemma 300M (Multilingual, Prefixes required)"
    },
    "embeddinggemma-2": {
        "id": "mlx-community/embeddinggemma-2-bf16",
        "type": "embedding_gemma2",
        "matryoshka": True,
        "supports_images": True,
        "description": "EmbeddingGemma 2 (Multilingual text + image, 768d, Prefixes required)"
    },
    # 2026-11-24 HFキャッシュを削除したため無効化中 (BENCHMARK_REPORT.md §9 で gemma 系に精度・速度とも劣後)。
    # 戻す場合は tests/test_embeddings.py の EMBED_MODELS / STANDARD_MODELS と
    # tests/data/test_cases.json の health.expected_embed_models も戻すこと:
    # "bge-m3": {
    #     "id": "mlx-community/bge-m3-mlx-fp16",
    #     "type": "standard",
    #     "description": "BGE-M3 (Multilingual, High precision)"
    # },
    # "bge-m3-8bit": {
    #     "id": "mlx-community/bge-m3-mlx-8bit",
    #     "type": "standard",
    #     "description": "BGE-M3 8-bit (Multilingual, Quantized)"
    # },
    # HFキャッシュ未保有 (かつ torch/torchvision が必要) のため無効化中:
    # "qwen3-vl-embedding-2b": {
    #     "id": "mlx-community/Qwen3-VL-Embedding-2B-mxfp8",
    #     "type": "qwen3_vl_embed",
    #     "description": "Qwen3-VL Embedding 2B (Multimodal, mlx-embeddings)"
    # },
    # 2026-11-25 HFキャッシュを削除したため無効化中 (BENCHMARK_REPORT.md §9 / §12 で全タスク最下位、
    # 速度も最も遅い。BF16 にしても変わらない)。qwen3_embed の処理は残してあるので、戻す場合は
    # ここと tests/test_embeddings.py の EMBED_MODELS / QWEN3_EMBED_MODELS、
    # tests/data/test_cases.json の health.expected_embed_models を戻すこと:
    # "qwen3-0.6b-embed": {
    #     "id": "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
    #     "type": "qwen3_embed",
    #     "matryoshka": True,
    #     "description": "Qwen3 Embedding 0.6B (Text only, last-token pooling)"
    # }
}

# EmbeddingGemma 系のタスクプロンプト (config_sentence_transformers.json の query / document)
GEMMA_QUERY_PREFIX = "task: search result | query: "
GEMMA_DOC_PREFIX = "title: none | text: "
# Qwen3-Embedding のクエリ指示文 (config_sentence_transformers.json の query)。文書側は指示文なし
QWEN3_EMBED_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
# Qwen3-Reranker の公式書式 (モデルカードの prefix / suffix / format_instruction)。
# 以前の書式 (<|im_end|>・<Instruct>・<think></think> なし) は nDCG@10 で 1.1〜10.9 低く、
# 課題によっては埋め込みだけより悪かった (BENCHMARK_REPORT.md §11)。
QWEN3_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
    "<|im_start|>user\n"
)
QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# モデルカードは多言語でも英語の指示文を推奨している
QWEN3_RERANK_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
# 文書はこのトークン数で切り詰める (長い文書で時間とメモリが際限なく増えないように)。
# §11 の評価は 512。512 と 1024 の差は ±0.3 ポイント以内だった
RERANK_MAX_DOC_TOKENS = 1024
EMBED_MAX_LENGTH = 2048
EMBED_BATCH_SIZE = 16
# 画像 1 枚あたりのトークン数 (Gemma4 image processor の max_soft_tokens が受け付ける値)。
# 画像は縦横比を保ってこのトークン数いっぱいまで拡大・縮小されるので、増やすほど高解像度になる。
# 既定はモデルの既定値 280。文字の多いページは 560 で大きく改善する (BENCHMARK_REPORT.md §10)。
IMAGE_TOKEN_OPTIONS = (70, 140, 280, 560, 1120)
DEFAULT_IMAGE_TOKENS = 280

AVAILABLE_RERANK_MODELS = {
    "qwen3-0.6b": {
        "id": "mlx-community/Qwen3-Reranker-0.6B-mxfp8",
        "type": "qwen3",
        "description": "Qwen3 Reranker 0.6B (Generative, Yes/No scoring)"
    },
    # HFキャッシュ未保有 (かつ torch/torchvision が必要) のため無効化中:
    # "qwen3-vl-reranker-2b": {
    #     "id": "mlx-community/Qwen3-VL-Reranker-2B-mxfp8",
    #     "type": "qwen3_vl_rerank",
    #     "description": "Qwen3-VL Reranker 2B (Multimodal, mlx-embeddings)"
    # }
}

AVAILABLE_AUDIO_MODELS = {
    # 2026-11-25 HFキャッシュを削除したため無効化中 (BENCHMARK_REPORT.md §5: 既定の 1.7B より
    # 0.25 秒速いだけで精度が大きく劣る)。戻す場合は tests/data/test_cases.json の
    # health.expected_audio_models も戻すこと:
    # "qwen3-asr-0.6b-8bit": {
    #     "id": "mlx-community/Qwen3-ASR-0.6B-8bit",
    #     "type": "asr",
    #     "description": "Qwen3 ASR 0.6B 8-bit (Speech-to-Text, Fast)"
    # },
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

# embeddinggemma-2: JMTEB サブセットで bge-m3 と同等の精度、約 2.3 倍速く、768 次元で画像も扱える
# (BENCHMARK_REPORT.md §9)。bge-m3 は 2026-11-24 に登録から外したため、bge-m3 (1024 次元) で
# 作った既存インデックスは再埋め込みが必要。
DEFAULT_EMBED = "embeddinggemma-2"
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

    def _load_embedding_gemma2(self, repo_id: str):
        """mlx_vlm でモデル本体だけをロードし、前処理は torch 不要の部品で組み立てる。

        mlx_vlm.load() は AutoProcessor を経由し、EmbeddingGemma2 の動画 processor が
        torch/torchvision を要求するため使わない。テキストは tokenizer、画像は
        Gemma4ImageProcessorPil (PIL 実装) で前処理する。
        """
        if vlm_load_model is None:
            raise HTTPException(status_code=500, detail="mlx-vlm (embedding_gemma2 support) not available")
        from transformers import AutoTokenizer
        from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

        path = vlm_get_model_path(repo_id)
        model = vlm_load_model(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        ip_cfg = dict(_json.loads((path / "processor_config.json").read_text())["image_processor"])
        ip_cfg.pop("image_processor_type", None)
        image_processor = Gemma4ImageProcessorPil(**ip_cfg)
        return model, {"tokenizer": tokenizer, "image_processor": image_processor}

    def _load_embed_unlocked(self, name: str, now: float):
        if name in self.embed_cache:
            return
        config = AVAILABLE_EMBED_MODELS[name]
        print(f"Loading Embedding model: {config['id']}...")
        if config["type"] == "embedding_gemma2":
            loaded = self._load_embedding_gemma2(config["id"])
        else:
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

def _l2(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.where(norm == 0, 1e-10, norm)


def _tokenizer_of(processor):
    """mlx_embeddings は TokenizerWrapper を返すことがあるので素の tokenizer を取り出す。"""
    return getattr(processor, "_tokenizer", processor)


def _batched_text_embeds(model, tokenizer, texts: List[str], ids_kwarg: str = "input_ids") -> np.ndarray:
    """text_embeds (モデル自身のプーリング + L2) を EMBED_BATCH_SIZE ずつ計算する。"""
    out = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        enc = tokenizer(texts[i:i + EMBED_BATCH_SIZE], return_tensors="np", padding=True,
                        truncation=True, max_length=EMBED_MAX_LENGTH)
        result = model(**{ids_kwarg: mx.array(enc["input_ids"]),
                          "attention_mask": mx.array(enc["attention_mask"])})
        out.append(np.array(result.text_embeds.astype(mx.float32)))
    return np.concatenate(out, axis=0)


def _embedding_gemma2_image(model, parts: dict, image, image_tokens: int = DEFAULT_IMAGE_TOKENS) -> np.ndarray:
    """画像 1 枚 -> 768 次元。<bos><boi>{<image> x n}<eoi><eos> を組み立てて forward する
    (EmbeddingGemma2Processor.replace_image_token と同じ展開)。
    image_tokens は呼び出しごとに processor へ渡す (共有の processor の設定は書き換えない)。"""
    tok, ip = parts["tokenizer"], parts["image_processor"]
    pix = ip(images=[image], max_soft_tokens=image_tokens, return_tensors="np")
    n = int(np.asarray(pix["num_soft_tokens_per_image"]).reshape(-1)[0])
    enc = tok([f"{tok.boi_token}{tok.image_token * n}{tok.eoi_token}"], return_tensors="np")
    result = model(
        mx.array(enc["input_ids"]),
        attention_mask=mx.array(enc["attention_mask"]),
        pixel_values=mx.array(pix["pixel_values"]),
        image_position_ids=mx.array(pix["image_position_ids"]),
    )
    return np.array(result.text_embeds.astype(mx.float32))[0]


def _gemma_prefix(input_type: str) -> str:
    return GEMMA_QUERY_PREFIX if input_type == "query" else GEMMA_DOC_PREFIX


def compute_embeddings(texts: List[Any], model_name: str, input_type: str = "document",
                       instruction: Optional[str] = None, dimensions: Optional[int] = None,
                       image_tokens: Optional[int] = None) -> List[List[float]]:
    """texts の要素は str、または embeddinggemma-2 の場合のみ PIL.Image。"""
    (model, processor), config = manager.get_embed(model_name)
    vecs = _compute_embeddings_raw(model, processor, config, texts, input_type, instruction,
                                   image_tokens or DEFAULT_IMAGE_TOKENS)
    if dimensions:
        if dimensions > vecs.shape[1]:
            raise HTTPException(status_code=400, detail=f"dimensions={dimensions} exceeds {model_name}'s {vecs.shape[1]}")
        vecs = _l2(vecs[:, :dimensions])
    return vecs.tolist()


def _compute_embeddings_raw(model, processor, config, texts, input_type, instruction,
                            image_tokens: int = DEFAULT_IMAGE_TOKENS):
    mtype = config["type"]

    # EmbeddingGemma (300M): mean pooling -> dense -> L2 は mlx_embeddings の text_embeds。
    # gemma3_text の __call__ は引数名が 'inputs'。
    if mtype == "gemma":
        prefixed = [_gemma_prefix(input_type) + t for t in texts]
        return _batched_text_embeds(model, _tokenizer_of(processor), prefixed, ids_kwarg="inputs")

    # EmbeddingGemma 2: テキストはプロンプト付きでバッチ処理、画像は 1 枚ずつ。元の順序で返す。
    if mtype == "embedding_gemma2":
        out = [None] * len(texts)
        text_idx = [i for i, t in enumerate(texts) if isinstance(t, str)]
        if text_idx:
            prefixed = [_gemma_prefix(input_type) + texts[i] for i in text_idx]
            for i, v in zip(text_idx, _batched_text_embeds(model, processor["tokenizer"], prefixed)):
                out[i] = v
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                out[i] = _embedding_gemma2_image(model, processor, t, image_tokens)
                mx.clear_cache()
        return _l2(np.stack(out))

    # Qwen3-Embedding: last-token pooling (tokenizer が末尾に <|endoftext|> を付ける) は text_embeds。
    # クエリだけ "Instruct: ...\nQuery:" を前置する。
    if mtype == "qwen3_embed":
        if input_type == "query":
            inst = instruction or QWEN3_EMBED_DEFAULT_INSTRUCTION
            texts = [f"Instruct: {inst}\nQuery:{t}" for t in texts]
        return _batched_text_embeds(model, _tokenizer_of(processor), texts)

    # Qwen3-VL Embedding モデル
    if mtype == "qwen3_vl_embed":
        inst = instruction or (
            "Represent this sentence for retrieving relevant documents."
            if input_type == "query"
            else "Represent this document for retrieval."
        )
        inputs = [{"text": t, "instruction": inst} for t in texts]
        embeddings = model.process(inputs, processor=processor)
        return _l2(np.array(embeddings))

    # standard (BGE-M3): CLS トークン + L2。mlx_embeddings の xlm_roberta.text_embeds は
    # mean pooling なので使わず、last_hidden_state の先頭を取る。
    if mtype == "standard":
        output = emb_generate(model, processor, texts=texts)
        return _l2(np.array(output.last_hidden_state[:, 0, :].astype(mx.float32)))

    raise HTTPException(status_code=500, detail=f"Unknown embedding model type: {mtype}")

def _truncate_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens])


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
            # 公式書式。プロンプト全体を 1 回でトークン化する (部分ごとだと境目のトークンが変わる)。
            # 1 文書ずつ forward する (まとめて処理すると mxfp8 の数値差で P(yes) が ±0.03 ずれる)
            prompt = (
                f"{QWEN3_RERANK_PREFIX}<Instruct>: {instruction or QWEN3_RERANK_DEFAULT_INSTRUCTION}\n"
                f"<Query>: {query}\n<Document>: {_truncate_tokens(tokenizer, doc, RERANK_MAX_DOC_TOKENS)}"
                f"{QWEN3_RERANK_SUFFIX}"
            )
            input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
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

EmbInputItem = Union[str, Dict[str, Any]]


class EmbReq(BaseModel):
    model: Optional[str] = DEFAULT_EMBED
    # str / str のリスト / 画像オブジェクトを含むリスト。画像オブジェクトの形式:
    #   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    #   {"type": "image", "image": "data:image/...;base64,..." または生の base64}
    input: Union[EmbInputItem, List[EmbInputItem]]
    input_type: Optional[str] = "document"
    instruction: Optional[str] = None
    # Matryoshka: 先頭 N 次元に切り詰めて再正規化 (OpenAI の `dimensions` と同名)
    dimensions: Optional[int] = None
    # 画像 1 枚あたりのトークン数 (IMAGE_TOKEN_OPTIONS のいずれか)。省略時は DEFAULT_IMAGE_TOKENS。
    # 画像対応モデルのみ。リクエストに画像が無ければ使われない
    image_tokens: Optional[int] = None


def _image_payload(item: Dict[str, Any]) -> str:
    if item.get("type") == "image_url" or "image_url" in item:
        url = item.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
    elif item.get("type") == "image":
        url = item.get("image")
    else:
        raise HTTPException(status_code=400, detail="input objects must be images: "
                            "{'type': 'image_url', 'image_url': {'url': 'data:...'}} or {'type': 'image', 'image': '<base64>'}")
    if not isinstance(url, str) or not url:
        raise HTTPException(status_code=400, detail="image input has no data")
    return url


def _decode_image(data: str):
    """data URI または生の base64 だけを受け付ける (URL やファイルパスは SSRF / ローカルファイル読み出しになるので拒否)。"""
    import base64
    import binascii
    from PIL import Image, UnidentifiedImageError

    if data.startswith("data:"):
        header, _, data = data.partition(",")
        if ";base64" not in header:
            raise HTTPException(status_code=400, detail="image data URI must be base64-encoded")
    elif data.startswith(("http://", "https://", "file:", "/", ".")):
        raise HTTPException(status_code=400, detail="image URLs and file paths are not accepted; send a base64 data URI")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
        img.load()
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"invalid image data: {e}")
    return img.convert("RGB")


def _parse_embedding_input(req: EmbReq, config: Dict[str, Any], model_name: str) -> List[Any]:
    raw = req.input if isinstance(req.input, list) else [req.input]
    if not raw:
        raise HTTPException(status_code=400, detail="input must not be empty")
    items: List[Any] = []
    for x in raw:
        if isinstance(x, str):
            # 孤立サロゲート (絵文字がチャンク境界で分割された断片) はトークナイザが落ちるため U+FFFD に置換
            items.append(x.encode("utf-16", "surrogatepass").decode("utf-16", "replace"))
            continue
        payload = _image_payload(x)
        if not config.get("supports_images"):
            raise HTTPException(status_code=400, detail=f"Model {model_name} does not accept image inputs")
        items.append(_decode_image(payload))
    return items


@app.post("/v1/embeddings")
async def embeddings(req: EmbReq):
    import traceback
    try:
        model_name = req.model or DEFAULT_EMBED
        # モデルをロードする前に検証する (不正なリクエストで GB 単位のロードを起こさない)
        config = AVAILABLE_EMBED_MODELS.get(model_name)
        if config is None:
            raise HTTPException(status_code=400, detail=f"Unsupported embedding model: {model_name}")
        if req.dimensions is not None:
            if not config.get("matryoshka"):
                raise HTTPException(status_code=400, detail=f"Model {model_name} does not support `dimensions`")
            if req.dimensions <= 0:
                raise HTTPException(status_code=400, detail="dimensions must be a positive integer")
        if req.image_tokens is not None:
            if not config.get("supports_images"):
                raise HTTPException(status_code=400, detail=f"Model {model_name} does not support `image_tokens`")
            if req.image_tokens not in IMAGE_TOKEN_OPTIONS:
                raise HTTPException(status_code=400,
                                    detail=f"image_tokens must be one of {list(IMAGE_TOKEN_OPTIONS)}")
        items = _parse_embedding_input(req, config, model_name)
        vecs = compute_embeddings(items, model_name, req.input_type, req.instruction, req.dimensions,
                                  req.image_tokens)

        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
            "model": model_name,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Embedding error: {tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{tb}")

class RerankReq(BaseModel):
    model: Optional[str] = DEFAULT_RERANK
    query: str
    documents: List[str]
    top_k: int = 10
    # Cohere / Jina 互換の名前。指定されていれば top_k より優先する (0 = 全件)。
    # OpenWebUI の外部 Reranker は top_n=len(documents) だけを送り、返ってきた結果を
    # index 順に並べて documents と位置で突き合わせる。top_n を無視して 10 件で切ると、
    # 候補が 11 件以上のときスコアが別の文書に付く
    top_n: Optional[int] = None
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

    limit = req.top_n if req.top_n is not None else req.top_k
    if limit:
        ranked = ranked[: min(limit, len(ranked))]

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

@app.get("/v1/models")
def list_models():
    data = []
    for kind, models in (
        ("embedding", AVAILABLE_EMBED_MODELS),
        ("rerank", AVAILABLE_RERANK_MODELS),
        ("audio", AVAILABLE_AUDIO_MODELS),
    ):
        for name in models:
            data.append({"id": name, "object": "model", "created": 0, "owned_by": "mlx-embed-rerank", "type": kind})
    return {"object": "list", "data": data}

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
# 監視スクリプト (run_mlx_server.sh) は 1236 を見る。並行して試験用サーバーを立てる場合のみ上書きする
HEALTH_PORT = int(os.environ.get("EMBED_HEALTH_PORT", "1236"))

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
