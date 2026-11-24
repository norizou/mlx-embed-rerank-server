"""Extract the JDocQA (jinaai/jdocqa_beir) page images to data/jdocqa/pages/<corpus-id>.png.

Inputs (download once; results in BENCHMARK_REPORT.md §10):
  data/jdocqa/corpus.parquet   corpus-id, image
  data/jdocqa/queries.parquet  query-id, query
  data/jdocqa/qrels.parquet    query-id, corpus-id, score

  B=https://huggingface.co/datasets/jinaai/jdocqa_beir/resolve/main/default
  for f in corpus queries qrels; do
    curl -sSL --fail -o data/jdocqa/$f.parquet $B/$f/test-00000-of-00001.parquet
  done

Pipeline: prep_jdocqa.py -> ocr_jdocqa.py -> eval_docimage.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).parent
DATA = HERE / "data" / "jdocqa"
PAGES = DATA / "pages"


def main():
    PAGES.mkdir(parents=True, exist_ok=True)
    corpus = pq.read_table(DATA / "corpus.parquet").to_pandas()
    sizes = []
    for _, row in corpus.iterrows():
        out = PAGES / f"{row['corpus-id']}.png"
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        sizes.append(img.size)
        if not out.exists():
            img.save(out)
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])
    stats = {
        "pages": len(sizes),
        "width_median": int(np.median(w)),
        "height_median": int(np.median(h)),
        "landscape": int((w > h).sum()),
        "min_side_p10": int(np.percentile(np.minimum(w, h), 10)),
    }
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
