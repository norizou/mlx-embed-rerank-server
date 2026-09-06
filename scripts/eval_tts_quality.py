# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
# ]
# ///

"""TTS 品質評価: 発音精度（CER）と声の再現度（話者類似度）を測る。

`scripts/benchmark_tts.py` が速度・長さ・メモリを測るのに対し、本スクリプトは
**生成された音声の中身**を評価します。`tests/data/voices/` の実素材を使います。

2 つの指標:

1. **発音精度（CER）** — 話者の書き起こし（`transcripts/*_fixed.txt`、人手修正済み）を
   その話者の声で合成し、ASR で書き戻して元テキストとの文字誤り率を出す。
   低いほど「指定したテキストを正しく発音できている」。

2. **声の再現度（話者類似度）** — 参照音声と生成音声の話者埋め込みのコサイン類似度。
   Qwen3-TTS の speaker encoder（x-vector 相当）を全モデル共通の物差しとして使う。
   絶対値だけでは解釈できないため、次の 2 つの基準値を併記する:
     * **上限の目安**: 同一話者の別音声どうしの類似度
     * **下限の目安**: 別話者どうしの類似度

前提:
  - サーバーが `http://localhost:1235` で稼働していること
  - `ffmpeg`（m4a の読み込みに使用）

使い方:
  uv run --extra dev python scripts/eval_tts_quality.py --out-dir /tmp/tts_quality
"""

import argparse
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import httpx

BASE_URL = "http://localhost:1235"
ASR_MODEL = "qwen3-asr-1.7b-8bit"
VOICES = Path("tests/data/voices")

# 参照音声と、その話者の書き起こし（人手修正済み）。
# 書き起こしは音声全体をカバーしていないものもあるため、
# ref_text（Qwen3 の ICL 用）は参照音声を ASR して自前で得る。
SPEAKERS = {
    "char": {
        "ref_audio": VOICES / "source/char.m4a",
        "transcript": VOICES / "transcripts/char_speech_fixed.txt",
        "other_take": VOICES / "source/char_short.m4a",  # 同一話者の別音声
    },
    "gihren": {
        "ref_audio": VOICES / "source/Gillen.m4a",
        "transcript": VOICES / "transcripts/gihren_speech_fixed.txt",
        "other_take": None,
    },
    "namihei": {
        "ref_audio": VOICES / "source/namihei.m4a",
        "transcript": VOICES / "transcripts/nagai_narration_fixed.txt",
        "other_take": None,
    },
}

MODELS = [
    ("irodori-8bit",           {"model": "irodori-tts-v4.1-small-8bit"}),
    ("irodori-fp16",           {"model": "irodori-tts-v4.1-small-fp16"}),
    ("irodori-8bit-steps10",   {"model": "irodori-tts-v4.1-small-8bit", "num_steps": 10}),
    ("qwen3-0.6b",             {"model": "qwen3-tts-0.6b-base-8bit"}),
    ("qwen3-1.7b",             {"model": "qwen3-tts-1.7b-base-8bit"}),
]

# 合成させるテキストのおおよその長さ（文字）。長すぎると 1 回の生成が重くなる。
# 実際は直後の句点まで含めて切る（文の途中で切ると生成が不自然になるため）。
SYNTH_CHARS = 100


# ---------------------------------------------------------------- テキスト正規化

def normalize(text: str) -> str:
    """CER 用の正規化: NFKC、空白と句読点・記号を除去。

    ASR は句読点の打ち方が安定しないため、句読点の違いを誤りとして数えない。
    """
    text = unicodedata.normalize("NFKC", text)
    drop = set(" \t\n　、。，．,.!?！？「」『』（）()・…ー-—〜~:：;；")
    return "".join(c for c in text if c not in drop)


def cut_at_sentence(text: str, approx_chars: int) -> str:
    """approx_chars を超えた直後の句点までで切る。

    文の途中で切ると、モデルが余った尺を伸ばし音で埋めるなど不自然な生成を招く。
    """
    text = text.strip()
    if len(text) <= approx_chars:
        return text
    idx = text.find("。", approx_chars - 1)
    return text[: idx + 1] if idx != -1 else text[:approx_chars]


def collapse_runs(text: str, max_run: int = 2) -> str:
    """同一文字の連続を max_run 個までに圧縮する。

    伸ばし音（「ジーーー」等）を ASR が同一文字の長大な連続として書き起こすため、
    素の CER だと発音の誤りではなく末尾アーティファクトが支配してしまう。
    圧縮版の CER は「テキストを正しく発音できたか」に近い量になる。
    """
    out, run, prev = [], 0, None
    for c in text:
        run = run + 1 if c == prev else 1
        prev = c
        if run <= max_run:
            out.append(c)
    return "".join(out)


def cer(reference: str, hypothesis: str) -> float:
    """文字誤り率 = レーベンシュタイン距離 / 参照文字数。"""
    r, h = normalize(reference), normalize(hypothesis)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


# ---------------------------------------------------------------- 音声ユーティリティ

def wav_seconds(path: Path) -> float:
    import wave
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def load_audio_24k(path: Path):
    """任意フォーマットを 24kHz mono float32 の mx.array で読む（話者埋め込み用）。"""
    import mlx.core as mx
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", "24000", "-"],
        capture_output=True, check=True,
    ).stdout
    return mx.array(np.frombuffer(raw, dtype=np.float32))


class SpeakerScorer:
    """Qwen3-TTS の speaker encoder を全モデル共通の物差しとして使う。"""

    def __init__(self):
        from mlx_audio.tts.utils import load_model
        print("話者埋め込み用モデルをロード中 (Qwen3-TTS-0.6B)...", flush=True)
        self.model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
        if getattr(self.model, "speaker_encoder", None) is None:
            raise RuntimeError("speaker_encoder を持たないモデルです")

    def embed(self, path: Path):
        import mlx.core as mx
        emb = self.model.extract_speaker_embedding(load_audio_24k(path), sr=24000)
        return emb / mx.linalg.norm(emb)

    def similarity(self, a: Path, b: Path) -> float:
        import mlx.core as mx
        return float(mx.sum(self.embed(a) * self.embed(b)).item())


# ---------------------------------------------------------------- サーバー呼び出し

def transcribe(client: httpx.Client, path: Path) -> str:
    with open(path, "rb") as fh:
        resp = client.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": (path.name, fh, "application/octet-stream")},
            data={"model": ASR_MODEL, "language": "ja"},
        )
    resp.raise_for_status()
    return resp.json()["text"]


def synthesize(client: httpx.Client, payload: dict, dest: Path) -> None:
    resp = client.post(f"{BASE_URL}/v1/audio/speech", json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    dest.write_bytes(resp.content)


# ---------------------------------------------------------------- 本体

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/tmp/tts_quality",
                    help="生成音声と結果 JSON の出力先（試聴用に残す）")
    ap.add_argument("--speakers", nargs="*", default=list(SPEAKERS))
    ap.add_argument("--chars", type=int, default=SYNTH_CHARS)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.speakers:
        if name not in SPEAKERS:
            print(f"エラー: 未知の話者 {name}", file=sys.stderr)
            sys.exit(1)

    scorer = SpeakerScorer()
    results, baselines = [], {}

    with httpx.Client(timeout=1800.0) as client:
        # --- 参照音声の ASR（Qwen3 の ref_text にする。音声と必ず一致する）---
        ref_texts = {}
        for name in args.speakers:
            src = SPEAKERS[name]["ref_audio"]
            ref_texts[name] = transcribe(client, src)
            print(f"[{name}] 参照 ASR: {ref_texts[name][:60]}...", flush=True)

        # --- 基準値: 同一話者の別テイク / 別話者 ---
        print("\n=== 話者類似度の基準値 ===", flush=True)
        for name in args.speakers:
            spec = SPEAKERS[name]
            if spec["other_take"] and spec["other_take"].exists():
                s = scorer.similarity(spec["ref_audio"], spec["other_take"])
                baselines[f"{name}: 同一話者の別音声"] = round(s, 4)
                print(f"  {name} vs 同一話者の別音声: {s:.4f}", flush=True)
        names = list(args.speakers)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s = scorer.similarity(SPEAKERS[names[i]]["ref_audio"],
                                      SPEAKERS[names[j]]["ref_audio"])
                baselines[f"{names[i]} vs {names[j]}（別話者）"] = round(s, 4)
                print(f"  {names[i]} vs {names[j]} (別話者): {s:.4f}", flush=True)

        # --- 本計測 ---
        for name in args.speakers:
            spec = SPEAKERS[name]
            target = cut_at_sentence(
                spec["transcript"].read_text(encoding="utf-8"), args.chars)
            print(f"\n=== 話者: {name} ===", flush=True)
            print(f"  合成テキスト({len(target)}文字): {target[:50]}...", flush=True)

            for label, extra in MODELS:
                payload = {
                    "input": target,
                    "ref_audio": str(spec["ref_audio"].resolve()),
                    "response_format": "wav",
                    **extra,
                }
                if payload["model"].startswith("qwen3"):
                    payload["ref_text"] = ref_texts[name]

                dest = out / f"{name}__{label}.wav"
                try:
                    synthesize(client, payload, dest)
                    hyp = transcribe(client, dest)
                    err = cer(target, hyp)
                    err_c = cer(collapse_runs(target), collapse_runs(hyp))
                    sim = scorer.similarity(spec["ref_audio"], dest)
                    dur = wav_seconds(dest)
                    results.append({
                        "speaker": name, "model": label,
                        "cer": round(err, 4), "cer_collapsed": round(err_c, 4),
                        "speaker_similarity": round(sim, 4),
                        "audio_sec": round(dur, 2),
                        "asr_text": hyp, "wav": str(dest),
                    })
                    print(f"  {label:22s} CER {err:7.2%} (圧縮後 {err_c:6.2%})  "
                          f"類似度 {sim:.4f}  音声長 {dur:5.2f}s", flush=True)
                except Exception as exc:
                    print(f"  {label:22s} FAILED: {str(exc)[:160]}",
                          file=sys.stderr, flush=True)
                    results.append({"speaker": name, "model": label,
                                    "error": str(exc)[:400]})

    payload_out = {
        "asr_model": ASR_MODEL,
        "synth_chars": args.chars,
        "note": "話者類似度は Qwen3-TTS の speaker encoder による。"
                "同エンジン由来のため Qwen3 側にわずかに有利な可能性がある。",
        "baselines": baselines,
        "ref_texts": ref_texts,
        "results": results,
    }
    (out / "results.json").write_text(
        json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n\n=== サマリ ===")
    print("| 話者 | モデル | CER | CER(圧縮後) | 話者類似度 | 音声長 |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for r in results:
        if "error" in r:
            print(f"| {r['speaker']} | {r['model']} | FAILED | | | |")
        else:
            print(f"| {r['speaker']} | {r['model']} | {r['cer']:.2%} | "
                  f"{r['cer_collapsed']:.2%} | {r['speaker_similarity']:.4f} | {r['audio_sec']:.2f}s |")
    print("\n基準値:")
    for k, v in baselines.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n音声と結果: {out}")


if __name__ == "__main__":
    main()
