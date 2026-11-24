"""Text-heavy image retrieval: JDocQA (jinaai/jdocqa_beir), question -> PDF page image.

758 page images (Japanese municipal reports, PR magazines, slides), 744 questions,
one relevant page per question (a few have 2-4).

Conditions
- image/t{N}:      embeddinggemma-2 embeds the whole page with an image token budget
                   of N (140/280/560/1120; the model default is 280). The processor
                   rescales the page, up or down, to fill the budget.
- image/tiles2x2:  page cut into 2x2 tiles (10% overlap), each at 280 tokens; the page
                   score is the best tile (same token total as t1120).
- ocr/<model>/head:   Apple Vision OCR text of the page, first 512 tokens (text embedder).
- ocr/<model>/chunks: OCR text split into ~400-char chunks; page score = best chunk.
- hybrid/a{w}:     w * image(best) + (1 - w) * ocr/embeddinggemma-2/chunks, both from
                   the same embeddinggemma-2 space.

Queries end with an answer-format instruction ("\n解答は…"); it is stripped for
every condition, and image/t280 is also scored with the raw query.

usage: python eval_docimage.py [--skip-tiles]
prereq: prep_jdocqa.py (page images), ocr_jdocqa.py (data/jdocqa/ocr.json)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from encoders import cache_path, load_encoder, peak_gb

ROOT = Path(__file__).parent
DATA = ROOT / "data/jdocqa"
RESULTS = ROOT / "results"
TOKEN_BUDGETS = (140, 280, 560, 1120)
CHUNK_CHARS = 400
HYBRID_WEIGHTS = (0.3, 0.5, 0.7)


def load():
    queries = pq.read_table(DATA / "queries.parquet").to_pandas()
    qrels_df = pq.read_table(DATA / "qrels.parquet").to_pandas()
    corpus_ids = sorted(int(p.stem) for p in (DATA / "pages").glob("*.png"))
    rel = {}
    for r in qrels_df.itertuples():
        rel.setdefault(int(r[1]), set()).add(int(r[2]))
    queries = queries[queries["query-id"].isin(rel)]
    raw = queries["query"].tolist()
    stripped = [re.sub(r"\n\s*解答は.*$", "", q, flags=re.S).strip() for q in raw]
    qrels = [rel[int(i)] for i in queries["query-id"]]
    pages = [str(DATA / "pages" / f"{c}.png") for c in corpus_ids]
    return raw, stripped, qrels, corpus_ids, pages


def metrics(S: np.ndarray, qrels, corpus_ids) -> dict:
    ids = np.array(corpus_ids)
    top = ids[np.argsort(-S, axis=1)[:, :10]]
    out = {k: [] for k in ("ndcg_10", "mrr_10", "R@1", "R@5", "R@10")}
    for row, rel in zip(top, qrels):
        hits = np.array([d in rel for d in row])
        ideal = (1 / np.log2(np.arange(2, 2 + min(len(rel), 10)))).sum()
        out["ndcg_10"].append((hits / np.log2(np.arange(2, 12))).sum() / ideal)
        out["mrr_10"].append(1 / (np.argmax(hits) + 1) if hits.any() else 0)
        for k in (1, 5, 10):
            out[f"R@{k}"].append(hits[:k].sum() / len(rel))
    return {k: round(float(np.mean(v)) * 100, 2) for k, v in out.items()}


def cached(model: str, key: str, fn):
    p = cache_path(model, key)
    if p.exists():
        return np.load(p)
    arr = fn()
    np.save(p, arr)
    return arr


def tiles(path: str, overlap: float = 0.1) -> list[Image.Image]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tw, th = w / 2, h / 2
    ox, oy = tw * overlap, th * overlap
    boxes = [(0, 0, tw + ox, th + oy), (tw - ox, 0, w, th + oy),
             (0, th - oy, tw + ox, h), (tw - ox, th - oy, w, h)]
    return [img.crop(tuple(int(v) for v in b)) for b in boxes]


def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    out, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) > size:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip() or not out:
        out.append(cur)
    return out


def max_over_groups(S: np.ndarray, owner: np.ndarray, n_pages: int) -> np.ndarray:
    """S: (n_q, n_units) -> (n_q, n_pages), taking each page's best unit."""
    out = np.full((S.shape[0], n_pages), -np.inf, dtype=np.float32)
    for page in range(n_pages):
        out[:, page] = S[:, owner == page].max(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tiles", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    raw, queries, qrels, corpus_ids, pages = load()
    n = len(pages)
    res = {"n_queries": len(queries), "n_pages": n, "retrieval": {}, "speed": {}}
    print(f"queries {len(queries)}, pages {n}", flush=True)

    def record(key, S):
        res["retrieval"][key] = metrics(S, qrels, corpus_ids)
        print(key, res["retrieval"][key], flush=True)

    # --- embeddinggemma-2: images + OCR text --------------------------------------
    enc, _ = load_encoder("embeddinggemma-2")
    Q = cached(enc.name, "jdocqa-q-stripped", lambda: enc.encode(queries, "query"))
    Qraw = cached(enc.name, "jdocqa-q-raw", lambda: enc.encode(raw, "query"))
    img_scores = {}
    for budget in TOKEN_BUDGETS:
        enc.encode_images(pages[:1], max_soft_tokens=budget)  # warm-up
        t = time.time()
        I = cached(enc.name, f"jdocqa-img-t{budget}", lambda: enc.encode_images(pages, max_soft_tokens=budget))
        res["speed"][f"image_t{budget}_ms_per_page"] = round((time.time() - t) / n * 1000, 1)
        img_scores[f"t{budget}"] = Q @ I.T
        record(f"image/t{budget}", img_scores[f"t{budget}"])
        if budget == 280:
            record("image/t280/raw_query", Qraw @ I.T)
    if not args.skip_tiles:
        tile_owner = np.repeat(np.arange(n), 4)
        t = time.time()
        T = cached(enc.name, "jdocqa-img-tiles2x2-t280",
                   lambda: enc.encode_images([tl for p in pages for tl in tiles(p)], max_soft_tokens=280))
        res["speed"]["image_tiles2x2_ms_per_page"] = round((time.time() - t) / n * 1000, 1)
        img_scores["tiles2x2"] = max_over_groups(Q @ T.T, tile_owner, n)
        record("image/tiles2x2", img_scores["tiles2x2"])
        record("image/tiles2x2+full_t280", np.maximum(img_scores["tiles2x2"], img_scores["t280"]))
    res["peak_mem_gb_eg2"] = round(peak_gb(), 2)

    # OCR is read only now, so the image passes can run while ocr_jdocqa.py finishes
    ocr = json.loads((DATA / "ocr.json").read_text())
    ocr_texts = [ocr[str(c)] for c in corpus_ids]
    chunks, owner = [], []
    for i, t in enumerate(ocr_texts):
        cs = chunk(t)
        chunks += cs
        owner += [i] * len(cs)
    owner = np.array(owner)
    res["n_ocr_chunks"] = len(chunks)
    res["ocr_chars_median"] = int(np.median([len(t) for t in ocr_texts]))
    print(f"ocr chunks {len(chunks)}, chars median {res['ocr_chars_median']}", flush=True)

    txt = {}
    D = cached(enc.name, "jdocqa-ocr-head", lambda: enc.encode(ocr_texts, "document"))
    record("ocr/embeddinggemma-2/head", Q @ D.T)
    C = cached(enc.name, f"jdocqa-ocr-chunks{CHUNK_CHARS}", lambda: enc.encode(chunks, "document"))
    txt["eg2"] = max_over_groups(Q @ C.T, owner, n)
    record("ocr/embeddinggemma-2/chunks", txt["eg2"])

    best_img_key = max(img_scores, key=lambda k: res["retrieval"][f"image/{k}"]["ndcg_10"])
    res["hybrid_image"] = best_img_key
    for w in HYBRID_WEIGHTS:
        record(f"hybrid/a{w}", w * img_scores[best_img_key] + (1 - w) * txt["eg2"])
    del enc

    # --- embeddinggemma-300m: OCR text only (the best text model in §9) -------------
    enc, _ = load_encoder("embeddinggemma-300m")
    Q3 = cached(enc.name, "jdocqa-q-stripped", lambda: enc.encode(queries, "query"))
    D3 = cached(enc.name, "jdocqa-ocr-head", lambda: enc.encode(ocr_texts, "document"))
    record("ocr/embeddinggemma-300m/head", Q3 @ D3.T)
    C3 = cached(enc.name, f"jdocqa-ocr-chunks{CHUNK_CHARS}", lambda: enc.encode(chunks, "document"))
    record("ocr/embeddinggemma-300m/chunks", max_over_groups(Q3 @ C3.T, owner, n))

    (RESULTS / "docimage_jdocqa.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
