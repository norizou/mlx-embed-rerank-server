import math
import numpy as np
import mlx.core as mx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any
import uvicorn
import threading

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
    }
}

AVAILABLE_RERANK_MODELS = {
    "qwen3-0.6b": {
        "id": "mlx-community/Qwen3-Reranker-0.6B-mxfp8",
        "type": "qwen3",
        "description": "Qwen3 Reranker 0.6B (Generative, Yes/No scoring)"
    }
}

DEFAULT_EMBED = "gemma-3-300m"
DEFAULT_RERANK = "qwen3-0.6b"

# =====================
# モデルマネージャ
# =====================

class ModelManager:
    def __init__(self):
        self.embed_cache = {}
        self.rerank_cache = {}
        self.lock = threading.Lock()

    def get_embed(self, name: str):
        if name not in AVAILABLE_EMBED_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported embedding model: {name}")
        
        with self.lock:
            if name not in self.embed_cache:
                config = AVAILABLE_EMBED_MODELS[name]
                print(f"Loading Embedding model: {config['id']}...")
                self.embed_cache[name] = emb_load(config['id'])
            return self.embed_cache[name], AVAILABLE_EMBED_MODELS[name]

    def get_rerank(self, name: str):
        if name not in AVAILABLE_RERANK_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported rerank model: {name}")
        
        with self.lock:
            if name not in self.rerank_cache:
                config = AVAILABLE_RERANK_MODELS[name]
                print(f"Loading Reranker model: {config['id']}...")
                model, tokenizer = lm_load(config['id'])
                
                # 特殊トークンの事前取得
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
            return self.rerank_cache[name], AVAILABLE_RERANK_MODELS[name]

manager = ModelManager()

# =====================
# 推論ロジック
# =====================

def compute_embeddings(texts: List[str], model_name: str, input_type: str = "document") -> List[List[float]]:
    (model, processor), config = manager.get_embed(model_name)
    
    # プレフィックス処理
    if config["type"] == "gemma":
        prefix = "task: search result | query: " if input_type == "query" else "title: none | text: "
        texts = [f"{prefix}{t}" for t in texts]
    
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

def compute_rerank(query: str, docs: List[str], model_name: str) -> List[float]:
    cached, config = manager.get_rerank(model_name)
    model = cached["model"]
    tokenizer = cached["tokenizer"]

    scores = []
    for doc in docs:
        if config["type"] == "qwen3":
            prompt = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. "
                "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                "<|im_start|>user\n"
                f"<Query>: {query}\n<Document>: {doc}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
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

@app.post("/v1/embeddings")
async def embeddings(req: EmbReq):
    import traceback
    try:
        texts = [req.input] if isinstance(req.input, str) else req.input
        model_name = req.model or DEFAULT_EMBED
        vecs = compute_embeddings(texts, model_name, req.input_type)
        
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

@app.post("/v1/rerank")
@app.post("/rerank")
async def rerank(req: RerankReq):
    model_name = req.model or DEFAULT_RERANK
    scores = compute_rerank(req.query, req.documents, model_name)

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
        "available_embed": list(AVAILABLE_EMBED_MODELS.keys()),
        "available_rerank": list(AVAILABLE_RERANK_MODELS.keys()),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1235)
