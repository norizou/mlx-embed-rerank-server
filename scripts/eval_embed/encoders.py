"""Model wrappers for the embedding evaluation.

Every encoder returns L2-normalised float32 numpy arrays so that cosine
similarity is a plain dot product.

- bge-m3:              mlx_embeddings, CLS pooling (same as the 1235 server)
- embeddinggemma-2:    mlx_vlm, mean pooling (model's own `text_embeds`)
- embeddinggemma-300m: mlx_vlm gemma3_embedding, the model's own pooling + dense
                       (the 1235 server instead takes the CLS token; this lets us
                       check whether that path under-reports the model)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# All weights come from the Hugging Face cache (same repos as the 1235 server's registry).
EG2_PATH = "mlx-community/embeddinggemma-2-bf16"
EG300_PATH = "mlx-community/embeddinggemma-300m-bf16"  # = server's gemma-3-300m (tensors are F16)
BGE_ID = "mlx-community/bge-m3-mlx-fp16"

GEMMA_QUERY = "task: search result | query: "
GEMMA_DOC = "title: none | text: "
GEMMA_STS = "task: sentence similarity | query: "

MAX_LEN = 512
BATCH = 16


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n == 0, 1e-10, n)


class Encoder:
    name: str
    supports_images = False

    def prefix(self, kind: str) -> str:
        return ""

    def _forward(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode(self, texts: list[str], kind: str = "document", batch: int = BATCH) -> np.ndarray:
        """kind: query | document | sts"""
        pre = self.prefix(kind)
        texts = [pre + t for t in texts]
        # sort by length so each batch pads little
        order = np.argsort([len(t) for t in texts])[::-1]
        chunks = []
        for i in range(0, len(texts), batch):
            idx = order[i : i + batch]
            chunks.append((idx, self._forward([texts[j] for j in idx])))
            mx.clear_cache()
        dim = chunks[0][1].shape[1]
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for idx, vecs in chunks:
            out[idx] = vecs
        return out


class BgeM3(Encoder):
    name = "bge-m3"

    def __init__(self):
        from mlx_embeddings.utils import load

        self.model, self.tok = load(BGE_ID)
        self.tok = getattr(self.tok, "_tokenizer", self.tok)

    def _forward(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
        out = self.model(mx.array(enc["input_ids"]), attention_mask=mx.array(enc["attention_mask"]))
        cls = out.last_hidden_state[:, 0, :].astype(mx.float32)
        return _normalize(np.array(cls))


class _MlxVlmEmbedder(Encoder):
    path: str

    def __init__(self):
        from mlx_vlm import load

        self.model, self.proc = load(self.path)
        self.tok = getattr(self.proc, "tokenizer", self.proc)

    def prefix(self, kind):
        return {"query": GEMMA_QUERY, "document": GEMMA_DOC, "sts": GEMMA_STS, "none": ""}[kind]

    def _forward(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
        out = self.model(mx.array(enc["input_ids"]), attention_mask=mx.array(enc["attention_mask"]))
        return _normalize(np.array(out.text_embeds.astype(mx.float32)))


class EmbeddingGemma2(_MlxVlmEmbedder):
    name = "embeddinggemma-2"
    path = EG2_PATH
    supports_images = True

    def encode_images(self, paths: list, bos: bool = True, prefix: str = "",
                      max_soft_tokens: int | None = None) -> np.ndarray:
        """One image per forward pass (as mlx_vlm's own server does).

        paths: file paths / URLs / PIL images.
        bos:   mlx_vlm's server calls prepare_inputs with add_special_tokens=False
               (no <bos>), unlike the text path; eval_image.py measures both.
        prefix: optional text placed before the image placeholder.
        max_soft_tokens: image token budget (70/140/280/560/1120; model default 280).
               The image processor rescales every image, up or down, to fill it.
        """
        from mlx_vlm.prompt_utils import get_chat_template
        from mlx_vlm.utils import load_image, prepare_inputs

        content = ([{"type": "text", "text": prefix}] if prefix else []) + [{"type": "image"}]
        messages = [{"role": "user", "content": content}]
        prompt = get_chat_template(self.proc, messages, add_generation_prompt=True)
        ip = self.proc.image_processor
        default_tokens = ip.max_soft_tokens
        if max_soft_tokens:
            ip.max_soft_tokens = max_soft_tokens
        try:
            vecs = []
            for p in paths:
                img = p if hasattr(p, "convert") else load_image(p)
                inputs = prepare_inputs(self.proc, images=[img], prompts=prompt, add_special_tokens=bos)
                out = self.model(**inputs)
                vecs.append(np.array(out.text_embeds.astype(mx.float32))[0])
                mx.clear_cache()
        finally:
            ip.max_soft_tokens = default_tokens
        return _normalize(np.stack(vecs))


class EmbeddingGemma300m(Encoder):
    """mlx_embeddings gemma3_text: `text_embeds` = mean pool -> dense -> L2 (the model's spec)."""

    name = "embeddinggemma-300m"
    server_cls = False

    def __init__(self):
        from mlx_embeddings.utils import load

        self.model, self.tok = load(EG300_PATH)
        self.tok = getattr(self.tok, "_tokenizer", self.tok)

    def prefix(self, kind):
        if self.server_cls:  # the server has no STS prompt
            return GEMMA_QUERY if kind == "query" else GEMMA_DOC
        return {"query": GEMMA_QUERY, "document": GEMMA_DOC, "sts": GEMMA_STS}[kind]

    def _forward(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
        out = self.model(inputs=mx.array(enc["input_ids"]), attention_mask=mx.array(enc["attention_mask"]))
        if self.server_cls:
            return _normalize(np.array(out.last_hidden_state[:, 0, :].astype(mx.float32)))
        return _normalize(np.array(out.text_embeds.astype(mx.float32)))


class EmbeddingGemma300mServerCLS(EmbeddingGemma300m):
    """Reproduces the 1235 server's `gemma` path: CLS token of last_hidden_state, no dense."""

    name = "embeddinggemma-300m-cls"
    server_cls = True


QWEN3_PATH = "mlx-community/Qwen3-Embedding-0.6B-mxfp8"  # = server's qwen3-0.6b-embed
QWEN3_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
QWEN3_STS = "Instruct: Retrieve semantically similar text.\nQuery:"


class Qwen3Embed(Encoder):
    """mlx_embeddings qwen3: `text_embeds` = last-token pool (tokenizer appends <|endoftext|>) -> L2.
    Query instruction from the model's config_sentence_transformers.json."""

    name = "qwen3-0.6b-embed"
    server_cls = False

    def __init__(self):
        from mlx_embeddings.utils import load

        self.model, self.tok = load(QWEN3_PATH)
        self.tok = getattr(self.tok, "_tokenizer", self.tok)

    def prefix(self, kind):
        if self.server_cls:  # the server sends raw text for `standard` models
            return ""
        return {"query": QWEN3_QUERY, "document": "", "sts": QWEN3_STS, "none": ""}[kind]

    def _forward(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
        out = self.model(mx.array(enc["input_ids"]), attention_mask=mx.array(enc["attention_mask"]))
        if self.server_cls:
            return _normalize(np.array(out.last_hidden_state[:, 0, :].astype(mx.float32)))
        return _normalize(np.array(out.text_embeds.astype(mx.float32)))


class Qwen3EmbedServerCLS(Qwen3Embed):
    """Reproduces the 1235 server's `standard` path for this model: CLS token, no instruction."""

    name = "qwen3-0.6b-embed-cls"
    server_cls = True


ENCODERS = {c.name: c for c in (BgeM3, EmbeddingGemma2, EmbeddingGemma300m, EmbeddingGemma300mServerCLS,
                                Qwen3Embed, Qwen3EmbedServerCLS)}


def load_encoder(name: str) -> tuple[Encoder, float]:
    mx.reset_peak_memory()
    t = time.time()
    enc = ENCODERS[name]()
    return enc, time.time() - t


def peak_gb() -> float:
    return mx.get_peak_memory() / 1e9


def cache_path(model: str, key: str) -> Path:
    p = Path(__file__).parent / "cache" / model / f"{key}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cached_encode(enc: Encoder, key: str, texts: list[str], kind: str) -> np.ndarray:
    p = cache_path(enc.name, key)
    if p.exists():
        arr = np.load(p)
        if arr.shape[0] == len(texts):
            return arr
    arr = enc.encode(texts, kind)
    np.save(p, arr)
    return arr


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
