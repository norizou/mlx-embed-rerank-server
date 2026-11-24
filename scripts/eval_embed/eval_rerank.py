"""Two-stage retrieval: embedding top-K -> qwen3-0.6b reranker (BENCHMARK_REPORT.md §11).

First stage reuses the embeddings cached by eval_text.py / eval_docimage.py:
  - JMTEB subsets (jagovfaqs_22k, jaqket, miracl-retrieval): embeddinggemma-2 and
    embeddinggemma-300m, same queries / 10k-doc corpora as §9.
  - JDocQA (§10): question -> page; first stage is the image(2x2 tiles)+OCR hybrid
    (w=0.5) and OCR chunks alone; the reranker reads the page's OCR text.
Second stage: Qwen3-Reranker-0.6B (mxfp8, the server's weights), score = P(yes) from
a softmax over the yes/no logits, in two prompt formats:
  - server:   the 1235 server's prompt (no <|im_end|>, no <Instruct>, no <think> block)
  - official: the model card's format (prefix/suffix, "<Instruct>: ...", <think></think>)
Documents are truncated to DOC_TOKENS (decoded back to text) for both formats.

Batching: right-padded batches; a causal LM's state at a sequence's own last real
token is unaffected by the padding after it, so we gather each row's last hidden
state and project it onto the yes/no rows of the output embedding only.

usage: python eval_rerank.py [--tasks ...] [--formats server official] [--k 50]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from encoders import cache_path
from eval_text import load_retrieval

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RERANKER = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"  # = server's qwen3-0.6b
# --reranker bf16: the official unquantised weights the mxfp8 one was made from
RERANKERS = {"mxfp8": RERANKER, "bf16": "Qwen/Qwen3-Reranker-0.6B"}
DOC_TOKENS = 512  # same budget as the first-stage embeddings (§9)
JDOCQA_QUERIES = 300  # sampled (seed 42) to match the JMTEB tasks; the full 744 take ~2x as long
TOKENS_PER_BATCH = 16_384
DEPTHS = (10, 20, 50)

SERVER_SYSTEM = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. "
                 "Note that the answer can only be \"yes\" or \"no\".\n<|im_start|>user\n")
OFFICIAL_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                   "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                   "<|im_start|>user\n")
OFFICIAL_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
OFFICIAL_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Reranker:
    def __init__(self, repo: str = RERANKER):
        from mlx_lm import load

        self.model, wrapper = load(repo)
        self.tok = getattr(wrapper, "_tokenizer", wrapper)
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        inner = self.model.model
        if self.model.args.tie_word_embeddings:
            rows = inner.embed_tokens(mx.array([self.no, self.yes]))  # dequantised rows
        else:
            rows = self.model.lm_head.weight[mx.array([self.no, self.yes])]
        self.head = rows.astype(mx.float32)  # (2, hidden)

    def _ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def truncate(self, doc: str) -> str:
        ids = self._ids(doc)
        return doc if len(ids) <= DOC_TOKENS else self.tok.decode(ids[:DOC_TOKENS])

    def build(self, fmt: str, query: str, doc: str) -> list[int]:
        """Tokenise the whole prompt in one pass, as the server does: tokenising the
        pieces separately changes the ids at the seams (and the server's scores)."""
        if fmt == "server":
            text = f"{SERVER_SYSTEM}<Query>: {query}\n<Document>: {doc}\n<|im_start|>assistant\n"
        else:
            text = (f"{OFFICIAL_PREFIX}<Instruct>: {OFFICIAL_INSTRUCTION}\n<Query>: {query}\n"
                    f"<Document>: {doc}{OFFICIAL_SUFFIX}")
        return self._ids(text)

    def score(self, seqs: list[list[int]], batch_tokens: int = TOKENS_PER_BATCH) -> np.ndarray:
        """batch_tokens=1 scores one sequence per forward, exactly like the server. Batched
        forwards differ from it by mxfp8 kernel numerics (not padding): about ±0.03 in P(yes)."""
        # ascending length, so the newest member of a batch is always its longest
        order = list(np.argsort([len(s) for s in seqs]))
        batches, cur = [], []
        for k in order:
            if cur and (len(cur) + 1) * len(seqs[k]) > batch_tokens:
                batches.append(cur)
                cur = []
            cur.append(k)
        if cur:
            batches.append(cur)
        out = np.zeros(len(seqs), dtype=np.float32)
        for idx in batches:
            width = len(seqs[idx[-1]])
            batch = np.zeros((len(idx), width), dtype=np.int32)
            last = np.zeros(len(idx), dtype=np.int32)
            for r, k in enumerate(idx):
                batch[r, : len(seqs[k])] = seqs[k]
                last[r] = len(seqs[k]) - 1
            h = self.model.model(mx.array(batch))  # (B, L, hidden), right padding is harmless
            h_last = h[mx.arange(len(idx)), mx.array(last)].astype(mx.float32)
            logits = h_last @ self.head.T  # (B, 2) = [no, yes]
            p_yes = mx.softmax(logits, axis=-1)[:, 1]
            out[idx] = np.array(p_yes)
            mx.clear_cache()
        return out


def metrics(order_ids: list[list[str]], qrels: list[set]) -> dict:
    out = {k: [] for k in ("ndcg_10", "mrr_10", "R@1", "R@5", "R@10")}
    for row, rel in zip(order_ids, qrels):
        hits = np.array([d in rel for d in row[:10]] + [False] * (10 - min(10, len(row))))
        ideal = (1 / np.log2(np.arange(2, 2 + min(len(rel), 10)))).sum()
        out["ndcg_10"].append((hits / np.log2(np.arange(2, 12))).sum() / ideal)
        out["mrr_10"].append(1 / (np.argmax(hits) + 1) if hits.any() else 0)
        for k in (1, 5, 10):
            out[f"R@{k}"].append(hits[:k].sum() / len(rel))
    return {k: round(float(np.mean(v)) * 100, 2) for k, v in out.items()}


def recall_at(order_ids, qrels, k) -> float:
    return round(float(np.mean([len(set(r[:k]) & rel) / len(rel) for r, rel in zip(order_ids, qrels)])) * 100, 2)


# ---------------------------------------------------------------- tasks
def jmteb_task(task: str):
    queries, qrels, docids, docs = load_retrieval(task)
    firsts = {}
    for model in ("embeddinggemma-2", "embeddinggemma-300m"):
        D = np.load(cache_path(model, f"{task}-corpus"))
        Q = np.load(cache_path(model, f"{task}-query"))
        firsts[model] = Q @ D.T
    return queries, qrels, docids, docs, firsts


def jdocqa_task():
    import eval_docimage as ed

    _raw, queries, qrels, corpus_ids, _pages = ed.load()
    ocr = json.loads((ed.DATA / "ocr.json").read_text())
    texts = [ocr[str(c)] for c in corpus_ids]
    chunks, owner = [], []
    for i, t in enumerate(texts):
        cs = ed.chunk(t)
        chunks += cs
        owner += [i] * len(cs)
    owner = np.array(owner)
    n = len(corpus_ids)
    m = "embeddinggemma-2"
    Q = np.load(cache_path(m, "jdocqa-q-stripped"))
    C = np.load(cache_path(m, f"jdocqa-ocr-chunks{ed.CHUNK_CHARS}"))
    T = np.load(cache_path(m, "jdocqa-img-tiles2x2-t280"))
    txt = ed.max_over_groups(Q @ C.T, owner, n)
    img = ed.max_over_groups(Q @ T.T, np.repeat(np.arange(n), 4), n)
    firsts = {"eg2-hybrid": 0.5 * img + 0.5 * txt, "eg2-ocr-chunks": txt}
    keep = sorted(random.Random(42).sample(range(len(queries)), min(JDOCQA_QUERIES, len(queries))))
    queries = [queries[i] for i in keep]
    qrels = [{str(d) for d in qrels[i]} for i in keep]
    firsts = {k: S[keep] for k, S in firsts.items()}
    return queries, qrels, [str(c) for c in corpus_ids], texts, firsts


# miracl-retrieval is supported but off by default: every model is above 98 nDCG@10 there (§9.7)
TASKS = ["jagovfaqs_22k", "jaqket", "jdocqa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--formats", nargs="+", default=["server", "official"])
    ap.add_argument("--k", type=int, default=max(DEPTHS))
    ap.add_argument("--single", action="store_true", help="one sequence per forward (= server numerics)")
    ap.add_argument("--reranker", choices=list(RERANKERS), default="mxfp8")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    tag = "" if args.reranker == "mxfp8" else f"-{args.reranker}"  # mxfp8 keeps the original file / cache names
    out_file = RESULTS / f"rerank_qwen3-0.6b{tag}.json"
    res = json.loads(out_file.read_text()) if out_file.exists() else {}
    rr = Reranker(RERANKERS[args.reranker])
    res["reranker"] = RERANKERS[args.reranker]
    res["doc_tokens"] = DOC_TOKENS
    for task in args.tasks:
        queries, qrels, docids, docs, firsts = jdocqa_task() if task == "jdocqa" else jmteb_task(task)
        tops = {name: np.argsort(-S, axis=1)[:, : args.k] for name, S in firsts.items()}
        need = sorted({(q, int(d)) for top in tops.values() for q in range(len(queries)) for d in top[q]})
        doc_txt = {d: rr.truncate(docs[d]) for d in {d for _, d in need}}
        tr = res.setdefault(task, {"n_queries": len(queries), "n_docs": len(docs), "pairs": len(need)})
        for name, top in tops.items():
            ids = [[docids[d] for d in row] for row in top]
            tr.setdefault(name, {})["first_stage"] = metrics(ids, qrels)
            tr[name]["first_stage"]["recall_ceiling"] = {f"R@{k}": recall_at(ids, qrels, k) for k in DEPTHS}
        for fmt in args.formats:
            label = f"{fmt}-single" if args.single else fmt
            key = cache_path(f"qwen3-0.6b-rerank{tag}", f"{task}-{label}-k{args.k}-d{DOC_TOKENS}")
            if key.exists():
                scores = dict(np.load(key, allow_pickle=True).item())
                elapsed = None
            else:
                t = time.time()
                seqs = [rr.build(fmt, queries[q], doc_txt[d]) for q, d in need]
                vals = rr.score(seqs, 1 if args.single else TOKENS_PER_BATCH)
                elapsed = time.time() - t
                scores = {pair: float(v) for pair, v in zip(need, vals)}
                np.save(key, np.array(scores, dtype=object))
                tr.setdefault("speed", {})[label] = {
                    "sec": round(elapsed, 1), "ms_per_pair": round(elapsed / len(need) * 1000, 2),
                    "avg_tokens": round(float(np.mean([len(s) for s in seqs])), 1)}
            for name, top in tops.items():
                for depth in DEPTHS:
                    reordered = []
                    for q, row in enumerate(top):
                        head = sorted(row[:depth], key=lambda d: -scores[(q, int(d))])
                        reordered.append([docids[d] for d in list(head) + list(row[depth:])])
                    tr[name][f"{label}/k{depth}"] = metrics(reordered, qrels)
                print(task, name, label, json.dumps({k: v for k, v in tr[name].items() if k.startswith(label + "/")}),
                      flush=True)
            out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
