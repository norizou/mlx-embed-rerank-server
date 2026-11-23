"""Japanese text-embedding evaluation on JMTEB subsets.

Retrieval tasks use a sub-sampled corpus (all positives of the sampled queries
+ random negatives up to CORPUS_SIZE). Absolute scores are therefore higher
than the official JMTEB leaderboard; the point is the model-to-model gap.

usage: python eval_text.py [--models bge-m3 embeddinggemma-2 ...] [--tasks ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from encoders import ENCODERS, cached_encode, load_encoder, peak_gb

ROOT = Path(__file__).parent
DATA = ROOT / "data/jmteb"
RESULTS = ROOT / "results"
SEED = 42
N_QUERIES = 300
CORPUS_SIZE = 10_000
MRL_DIMS = (768, 512, 256, 128)


def _rel(v) -> list[str]:
    v = ast.literal_eval(v) if isinstance(v, str) else v
    return [str(x) for x in v]


def load_retrieval(task: str):
    """returns queries[list[str]], qrels[list[set[docid]]], docids[list], docs[list[str]]"""
    rng = random.Random(SEED)
    q = pq.read_table(DATA / f"{task}-query/test.parquet").to_pylist()
    q = rng.sample(q, min(N_QUERIES, len(q)))
    queries = [r["query"] for r in q]
    qrels = [set(_rel(r["relevant_docs"])) for r in q]
    positives = set().union(*qrels)

    corpus_file = DATA / f"{task}-corpus/corpus.parquet"
    ids = pq.read_table(corpus_file, columns=["docid"]).column("docid").to_pylist()
    ids = [str(i) for i in ids]
    negatives = [i for i in ids if i not in positives]
    keep = positives | set(rng.sample(negatives, CORPUS_SIZE - len(positives)))

    # jaqket's text already starts with the article title, so title is not prepended
    tbl = pq.read_table(corpus_file, columns=["docid", "text"])
    tbl = tbl.filter(pc.is_in(pc.cast(tbl["docid"], "string"), value_set=pa.array(sorted(keep))))
    rows = tbl.to_pylist()
    docids = [str(r["docid"]) for r in rows]
    docs = [r["text"] for r in rows]
    missing = positives - set(docids)
    assert not missing, f"{task}: {len(missing)} positives missing from corpus"
    return queries, qrels, docids, docs


def retrieval_metrics(S: np.ndarray, qrels, docids) -> dict:
    ids = np.array(docids)
    top = np.argsort(-S, axis=1)[:, :100]
    ndcg, mrr, r10, r100 = [], [], [], []
    for row, rel in zip(top, qrels):
        hits = np.array([ids[j] in rel for j in row])
        gains = hits[:10] / np.log2(np.arange(2, 12))
        ideal = (1 / np.log2(np.arange(2, 2 + min(len(rel), 10)))).sum()
        ndcg.append(gains.sum() / ideal)
        first = np.argmax(hits[:10]) if hits[:10].any() else None
        mrr.append(0 if first is None else 1 / (first + 1))
        r10.append(hits[:10].sum() / len(rel))
        r100.append(hits.sum() / len(rel))
    return {k: round(float(np.mean(v)) * 100, 2) for k, v in
            dict(ndcg_10=ndcg, mrr_10=mrr, recall_10=r10, recall_100=r100).items()}


def truncate(E: np.ndarray, d: int) -> np.ndarray:
    E = E[:, :d]
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def run_retrieval(enc, task: str) -> dict:
    queries, qrels, docids, docs = load_retrieval(task)
    t = time.time()
    D = cached_encode(enc, f"{task}-corpus", docs, "document")
    Q = cached_encode(enc, f"{task}-query", queries, "query")
    res = {"n_queries": len(queries), "n_docs": len(docs), "encode_sec": round(time.time() - t, 1)}
    dims = MRL_DIMS if enc.name == "embeddinggemma-2" else (D.shape[1],)
    for d in dims:
        res[f"dim{d}"] = retrieval_metrics(truncate(Q, d) @ truncate(D, d).T, qrels, docids)
    return res


def run_jsts(enc) -> dict:
    rows = pq.read_table(DATA / "jsts/test.parquet").to_pylist()
    a = cached_encode(enc, "jsts-a", [r["sentence1"] for r in rows], "sts")
    b = cached_encode(enc, "jsts-b", [r["sentence2"] for r in rows], "sts")
    gold = [float(r["label"]) for r in rows]
    return {"n_pairs": len(rows), "spearman": round(float(spearmanr((a * b).sum(1), gold).statistic) * 100, 2)}


def run_speed(enc) -> dict:
    """1000 jagovfaqs docs, cold (first) and warm (second) pass."""
    docs = pq.read_table(DATA / "jagovfaqs_22k-corpus/corpus.parquet", columns=["text"]).column("text").to_pylist()[:1000]
    out = {}
    for label in ("cold", "warm"):
        t = time.time()
        enc.encode(docs, "document")
        dt = time.time() - t
        out[label] = {"sec": round(dt, 2), "ms_per_doc": round(dt, 3)}
    out["avg_chars"] = round(sum(map(len, docs)) / len(docs))
    return out


TASKS = ["jagovfaqs_22k", "jaqket", "miracl-retrieval", "jsts", "speed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ENCODERS))
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for name in args.models:
        enc, load_sec = load_encoder(name)
        out_file = RESULTS / f"text_{name}.json"
        res = json.loads(out_file.read_text()) if out_file.exists() else {}
        res["load_sec"] = round(load_sec, 1)
        for task in args.tasks:
            print(f"[{name}] {task} ...", flush=True)
            if task == "jsts":
                res[task] = run_jsts(enc)
            elif task == "speed":
                res[task] = run_speed(enc)
            else:
                res[task] = run_retrieval(enc, task)
            res["peak_mem_gb"] = round(peak_gb(), 2)
            print(f"[{name}] {task}: {json.dumps(res[task], ensure_ascii=False)}", flush=True)
            out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2))
        del enc


if __name__ == "__main__":
    main()
