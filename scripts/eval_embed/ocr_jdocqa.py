# /// script
# requires-python = ">=3.13"
# dependencies = ["ocrmac", "pillow"]
# ///
"""OCR the JDocQA pages with Apple's Vision framework (macOS only).

The OCR text shipped with jinaai/jdocqa (EasyOCR) is garbled for Japanese, so the
text baselines in eval_docimage.py use this instead. Lines are joined in Vision's
reading order. Output: data/jdocqa/ocr.json  {corpus-id: text}

Run (outside the sandbox; Vision needs system services):
  uv run scripts/eval_embed/ocr_jdocqa.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ocrmac import ocrmac

HERE = Path(__file__).parent
PAGES = HERE / "data" / "jdocqa" / "pages"
OUT = HERE / "data" / "jdocqa" / "ocr.json"


def main():
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    pages = sorted(PAGES.glob("*.png"), key=lambda p: int(p.stem))
    t0 = time.time()
    for i, p in enumerate(pages):
        if p.stem in done:
            continue
        res = ocrmac.OCR(str(p), language_preference=["ja-JP", "en-US"], recognition_level="accurate").recognize()
        done[p.stem] = "\n".join(text for text, _conf, _box in res)
        if i % 50 == 0:
            OUT.write_text(json.dumps(done, ensure_ascii=False))
            print(f"{i}/{len(pages)} {time.time() - t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False))
    lens = sorted(len(t) for t in done.values())
    print(f"done {len(done)} pages, chars median {lens[len(lens) // 2]}, empty {sum(n == 0 for n in lens)}")


if __name__ == "__main__":
    main()
