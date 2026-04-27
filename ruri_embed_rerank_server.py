from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Optional
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

# =====================
# 設定
# =====================
EMBED_MODEL_ID = "cl-nagoya/ruri-v3-70m"
RERANK_MODEL_ID = "cl-nagoya/ruri-v3-reranker-310m"

device = "mps" if torch.backends.mps.is_available() else "cpu"

app = FastAPI(title="Ruri Embedding + Rerank Server")

# =====================
# Embedding 初期化
# =====================
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_ID).to(device)
embed_model.eval()

def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted

def embed_texts(texts: List[str]) -> List[List[float]]:
    with torch.no_grad():
        enc = embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = embed_model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32).tolist()

# =====================
# Reranker 初期化
# =====================
reranker = CrossEncoder(RERANK_MODEL_ID, device=device)

# =====================
# API定義
# =====================
class EmbReq(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]

@app.post("/v1/embeddings")
def embeddings(req: EmbReq):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = embed_texts(texts)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vecs)
        ],
        "model": req.model or EMBED_MODEL_ID,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

class RerankReq(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 10

def run_rerank(query: str, docs: list[str]) -> list[float]:
    """
    query と documents の CrossEncoder スコアを返す
    戻り値は documents と同じ順序のスコア配列
    """
    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)
    return [float(s) for s in scores]

def do_rerank(req: RerankReq):
    scores = run_rerank(req.query, req.documents)

    ranked = sorted(
        (
            {"index": i, "relevance_score": scores[i]}
            for i in range(len(req.documents))
        ),
        key=lambda x: x["relevance_score"],
        reverse=True,
    )

    if req.top_k:
        ranked = ranked[: min(req.top_k, len(ranked))]

    return {
        "model": RERANK_MODEL_ID,
        "results": ranked,
    }

@app.post("/v1/rerank")
def rerank_v1(req: RerankReq):
    return do_rerank(req)

@app.post("/rerank")
def rerank_alias(req: RerankReq):
    return do_rerank(req)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "embedding_model": EMBED_MODEL_ID,
        "rerank_model": RERANK_MODEL_ID,
    }
