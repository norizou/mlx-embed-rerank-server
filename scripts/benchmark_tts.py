# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
# ]
# ///

"""TTS ベンチマーク: Irodori v4.1 の fp16 / 8bit 比較と Qwen3-TTS の対照。

BENCHMARK_REPORT.md §6 と同じ手法（ウォームアップ後に 10 回連続実行）で、
生成時間・音声長・リアルタイム係数を計測します。あわせてモデルごとの
コールドロード時間と、サーバープロセスの RSS 増分（常駐コスト）を記録します。

前提:
  - サーバーが起動しており、**対象モデルが未ロード**であること
    （コールドロード時間を測るため。`./run_mlx_server.sh restart` 直後に実行）

使い方:
  uv run --extra dev python scripts/benchmark_tts.py \
      --ref-audio tests/data/voices/clips/namihei_10s.wav \
      --out /tmp/tts_bench.json
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

import httpx

BASE_URL = "http://localhost:1235"
RUNS = 10

# BENCHMARK_REPORT.md §6 と同一の入力テキスト（過去の計測と比較できるように）
INPUT_TEXT = (
    "人工知能と機械学習は、今日のソフトウェア開発において不可欠な技術となっています。"
    "モデルの精度を高めるためには、質の高いデータと継続的な評価が重要です。"
)
INSTRUCT = "落ち着いた男性の声で、ニュース原稿のように明瞭に読み上げてください。"


def server_rss_mb() -> float | None:
    """サーバープロセスの RSS(MB)。常駐コストの増分を見るために使う。"""
    try:
        pid = subprocess.run(
            ["lsof", "-ti:1235"], capture_output=True, text=True, timeout=10
        ).stdout.split()
        if not pid:
            return None
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", pid[0]], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return int(out) / 1024 if out else None
    except Exception:
        return None


def wav_duration_sec(data: bytes) -> float:
    with wave.open(BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def speak(client: httpx.Client, payload: dict) -> tuple[float, float, int]:
    """1 回生成し (経過秒, 音声長秒, バイト数) を返す。"""
    started = time.perf_counter()
    resp = client.post(f"{BASE_URL}/v1/audio/speech", json=payload)
    elapsed = time.perf_counter() - started
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return elapsed, wav_duration_sec(resp.content), len(resp.content)


def measure(client: httpx.Client, label: str, payload: dict, runs: int) -> dict:
    """コールドロード → ウォームアップ → runs 回の計測。"""
    print(f"\n=== {label} ===", flush=True)

    rss_before = server_rss_mb()
    cold_started = time.perf_counter()
    cold_elapsed, cold_audio, _ = speak(client, payload)
    cold_total = time.perf_counter() - cold_started
    rss_after_load = server_rss_mb()
    print(f"  cold (load+generate): {cold_total:.2f}s", flush=True)

    # ウォームアップ 1 回（MLX のコンパイルキャッシュを温める）
    speak(client, payload)
    rss_warm = server_rss_mb()

    gens, durs = [], []
    for i in range(runs):
        elapsed, dur, nbytes = speak(client, payload)
        gens.append(elapsed)
        durs.append(dur)
        print(f"  run {i + 1:2d}: {elapsed:6.2f}s / audio {dur:5.2f}s "
              f"/ RTF {dur / elapsed:4.2f}x", flush=True)

    mean_gen = statistics.fmean(gens)
    mean_dur = statistics.fmean(durs)
    return {
        "label": label,
        "model": payload["model"],
        "cold_total_sec": round(cold_total, 3),
        "cold_generate_sec": round(cold_elapsed, 3),
        "cold_audio_sec": round(cold_audio, 3),
        "runs": runs,
        "generate_sec": [round(x, 3) for x in gens],
        "audio_sec": [round(x, 3) for x in durs],
        "generate_mean": round(mean_gen, 3),
        "generate_std": round(statistics.stdev(gens), 3) if len(gens) > 1 else 0.0,
        "generate_min": round(min(gens), 3),
        "generate_max": round(max(gens), 3),
        "audio_mean": round(mean_dur, 3),
        "audio_std": round(statistics.stdev(durs), 3) if len(durs) > 1 else 0.0,
        "rtf": round(mean_dur / mean_gen, 2),
        "rss_before_mb": round(rss_before) if rss_before else None,
        "rss_after_load_mb": round(rss_after_load) if rss_after_load else None,
        "rss_warm_mb": round(rss_warm) if rss_warm else None,
        "rss_delta_mb": (
            round(rss_after_load - rss_before)
            if rss_before and rss_after_load else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-audio", required=True, help="参照音声（サーバーから見たパス）")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--out", default="/tmp/tts_bench.json")
    args = ap.parse_args()

    ref = str(Path(args.ref_audio).resolve())
    if not Path(ref).exists():
        print(f"エラー: 参照音声が見つかりません: {ref}", file=sys.stderr)
        sys.exit(1)

    with httpx.Client(timeout=900.0) as client:
        health = client.get(f"{BASE_URL}/health").json()
        loaded = health["loaded_tts_models"]
        print(f"開始時のロード済み TTS: {loaded or '(なし)'}")
        print(f"サーバー RSS: {server_rss_mb():.0f} MB")
        print(f"参照音声: {ref}")

        cases = [
            # Irodori v4.1: ボイスクローン（本命の比較）
            ("irodori-v4.1-8bit / clone", {
                "model": "irodori-tts-v4.1-small-8bit",
                "input": INPUT_TEXT, "ref_audio": ref, "response_format": "wav",
            }),
            ("irodori-v4.1-fp16 / clone", {
                "model": "irodori-tts-v4.1-small-fp16",
                "input": INPUT_TEXT, "ref_audio": ref, "response_format": "wav",
            }),
            # Irodori v4.1: VoiceDesign（caption のみ）
            ("irodori-v4.1-8bit / caption", {
                "model": "irodori-tts-v4.1-small-8bit",
                "input": INPUT_TEXT, "instruct": INSTRUCT, "response_format": "wav",
            }),
            ("irodori-v4.1-fp16 / caption", {
                "model": "irodori-tts-v4.1-small-fp16",
                "input": INPUT_TEXT, "instruct": INSTRUCT, "response_format": "wav",
            }),
            # Irodori v4.1: num_steps を下げた高速プリセット
            ("irodori-v4.1-8bit / clone num_steps=10", {
                "model": "irodori-tts-v4.1-small-8bit",
                "input": INPUT_TEXT, "ref_audio": ref, "num_steps": 10,
                "response_format": "wav",
            }),
            # 対照: 現行デフォルトの Qwen3-TTS（ICL には書き起こしが要る）
            ("qwen3-tts-0.6b-8bit / clone", {
                "model": "qwen3-tts-0.6b-base-8bit",
                "input": INPUT_TEXT, "ref_audio": ref,
                "ref_text": "人類が増えすぎた人口を宇宙に移民させるようになって、すでに半世紀。",
                "response_format": "wav",
            }),
        ]

        results = []
        for label, payload in cases:
            try:
                results.append(measure(client, label, payload, args.runs))
            except Exception as exc:  # 1 ケースの失敗で全体を落とさない
                print(f"  !! FAILED: {exc}", file=sys.stderr, flush=True)
                results.append({"label": label, "model": payload["model"],
                                "error": str(exc)[:500]})

        final = client.get(f"{BASE_URL}/health").json()

    payload_out = {
        "input_text": INPUT_TEXT,
        "instruct": INSTRUCT,
        "ref_audio": ref,
        "runs": args.runs,
        "loaded_tts_at_end": final["loaded_tts_models"],
        "server_rss_end_mb": round(server_rss_mb() or 0),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload_out, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print("\n\n=== サマリ ===")
    header = ("| ケース | コールド | 平均生成 | 生成std | 平均音声長 | 音声長std | RT係数 | RSS増分 |")
    print(header)
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        if "error" in r:
            print(f"| {r['label']} | FAILED | | | | | | |")
            continue
        print(f"| {r['label']} | {r['cold_total_sec']:.2f}s | {r['generate_mean']:.3f}s "
              f"| {r['generate_std']:.3f} | {r['audio_mean']:.3f}s | {r['audio_std']:.3f} "
              f"| {r['rtf']:.2f}x | {r['rss_delta_mb'] or '-'} MB |")
    print(f"\n結果を保存: {args.out}")


if __name__ == "__main__":
    main()
