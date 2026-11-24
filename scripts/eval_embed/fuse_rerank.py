"""Score fusion over the cached rerank scores (no model runs): within the first-stage
top-K, rank by  w * minmax(first-stage cosine) + (1 - w) * minmax(reranker P(yes)),
both min-max normalised per query. w = 0 is the reranker alone, w = 1 the first stage.

usage: python fuse_rerank.py [--format official] [--k 20 50]
prereq: eval_rerank.py (cache/qwen3-0.6b-rerank/*.npy)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import eval_rerank as er
from encoders import cache_path

WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)


def minmax(v: np.ndarray) -> np.ndarray:
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 0 else np.zeros_like(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="official")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 50])
    args = ap.parse_args()
    out_file = er.RESULTS / "rerank_qwen3-0.6b.json"
    res = json.loads(out_file.read_text())
    for task in er.TASKS:
        queries, qrels, docids, docs, firsts = er.jdocqa_task() if task == "jdocqa" else er.jmteb_task(task)
        key = cache_path("qwen3-0.6b-rerank", f"{task}-{args.format}-k{max(er.DEPTHS)}-d{er.DOC_TOKENS}")
        scores = dict(np.load(key, allow_pickle=True).item())
        for name, S in firsts.items():
            top = np.argsort(-S, axis=1)[:, : max(er.DEPTHS)]
            fused = {}
            for k in args.k:
                for w in WEIGHTS:
                    orders = []
                    for q, row in enumerate(top):
                        head = row[:k]
                        f = minmax(S[q, head])
                        r = minmax(np.array([scores[(q, int(d))] for d in head]))
                        mix = w * f + (1 - w) * r
                        orders.append([docids[d] for d in head[np.argsort(-mix)]])
                    fused[f"k{k}/w{w}"] = er.metrics(orders, qrels)
            res[task][name][f"fusion_{args.format}"] = fused
            best = max(fused, key=lambda x: fused[x]["ndcg_10"])
            print(task, name, "best", best, fused[best]["ndcg_10"],
                  "| w0.5:", {k: fused[f"k{k}/w0.5"]["ndcg_10"] for k in args.k}, flush=True)
    out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
