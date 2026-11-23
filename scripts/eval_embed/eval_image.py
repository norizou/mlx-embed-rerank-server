"""embeddinggemma-2 image-embedding evaluation on COCO 1k (Karpathy test, first 1000).

- Text->image / image->text Recall@1/5/10 with Japanese (STAIR Captions v1.2)
  and English (COCO) captions, 5 captions per image.
- Variants: image prompt with/without <bos>; captions with the retrieval query
  prompt vs no prompt.
- Sanity: augmented copy of the same image vs other images.
- Speed / peak memory per image.

usage: python eval_image.py [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from encoders import cache_path, load_encoder, peak_gb

ROOT = Path(__file__).parent
COCO = ROOT / "data/coco"
RESULTS = ROOT / "results"


def load_coco(n: int):
    df = pd.read_parquet(COCO / "karpathy_test.parquet").head(n)
    stair = defaultdict(list)
    for a in json.load(open(COCO / "stair_captions_v1.2_val.json"))["annotations"]:
        stair[a["image_id"]].append(a["caption"].strip())
    paths, ja, en = [], [], []
    for r in df.itertuples():
        p = COCO / "images" / r.filename
        try:
            Image.open(p).verify()
        except Exception:
            continue
        if len(stair[r.cocoid]) < 5:
            continue
        paths.append(str(p))
        ja.append(stair[r.cocoid][:5])
        en.append([s.strip() for s in r.sentences[:5]])
    return paths, ja, en


def recall(I: np.ndarray, T: np.ndarray, owner: np.ndarray) -> dict:
    """I: (n_img, d), T: (n_cap, d), owner[c] = image index of caption c."""
    S = T @ I.T  # (n_cap, n_img)
    rank_t2i = (S > S[np.arange(len(T)), owner][:, None]).sum(1)  # 0 = top
    # i2t: best rank among the image's own captions
    St = S.T
    best = np.full(len(I), np.inf)
    order = np.argsort(-St, axis=1)
    pos = np.empty_like(order)
    rows = np.arange(len(I))[:, None]
    pos[rows, order] = np.arange(St.shape[1])[None, :]
    for c, i in enumerate(owner):
        best[i] = min(best[i], pos[i, c])
    out = {}
    for k in (1, 5, 10):
        out[f"t2i_R@{k}"] = round(float((rank_t2i < k).mean()) * 100, 2)
        out[f"i2t_R@{k}"] = round(float((best < k).mean()) * 100, 2)
    out["rsum"] = round(sum(out.values()), 2)
    matched = S[np.arange(len(T)), owner]
    out["cos_matched"] = round(float(matched.mean()), 4)
    out["cos_unmatched"] = round(float((S.sum() - matched.sum()) / (S.size - len(T))), 4)
    return out


def cached(name: str, fn):
    p = cache_path("embeddinggemma-2", name)
    if p.exists():
        return np.load(p)
    arr = fn()
    np.save(p, arr)
    return arr


def augment(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
    return img.resize((int(img.width * 0.6), int(img.height * 0.6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    paths, ja, en = load_coco(args.n)
    n = len(paths)
    owner = np.repeat(np.arange(n), 5)
    print(f"images: {n}", flush=True)

    enc, load_sec = load_encoder("embeddinggemma-2")
    res = {"n_images": n, "n_captions_per_lang": n * 5, "load_sec": round(load_sec, 1)}

    # speed (first 50 images, warm after the first call)
    enc.encode_images(paths[:2])
    t = time.time()
    enc.encode_images(paths[:50])
    res["speed"] = {"ms_per_image": round((time.time() - t) / 50 * 1000, 1)}

    imgs = {
        "bos": cached(f"coco{n}-img-bos", lambda: enc.encode_images(paths, bos=True)),
        "nobos": cached(f"coco{n}-img-nobos", lambda: enc.encode_images(paths, bos=False)),
    }
    res["peak_mem_gb"] = round(peak_gb(), 2)
    caps = {}
    for lang, c in (("ja", ja), ("en", en)):
        flat = [s for five in c for s in five]
        for kind in ("query", "none"):
            caps[(lang, kind)] = cached(f"coco{n}-cap-{lang}-{kind}", lambda: enc.encode(flat, kind))

    res["retrieval"] = {}
    for img_key, I in imgs.items():
        for (lang, kind), T in caps.items():
            key = f"{lang}/img_{img_key}/cap_{kind}"
            res["retrieval"][key] = recall(I, T, owner)
            print(key, res["retrieval"][key], flush=True)

    # sanity: same image augmented vs other images (bos variant)
    rng = random.Random(0)
    sample = rng.sample(range(n), 50)
    A = enc.encode_images([augment(paths[i]) for i in sample], bos=True)
    I = imgs["bos"]
    full = A @ I.T
    others = full.copy()
    others[np.arange(50), sample] = np.nan
    res["sanity_augmented"] = {
        "cos_same_image": round(float(full[np.arange(50), sample].mean()), 4),
        "cos_other_images": round(float(np.nanmean(others)), 4),
        "top1_is_original_pct": round(float((full.argmax(1) == np.array(sample)).mean()) * 100, 1),
    }
    print("sanity", res["sanity_augmented"], flush=True)

    (RESULTS / "image_embeddinggemma-2.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
