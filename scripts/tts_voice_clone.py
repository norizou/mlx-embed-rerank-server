# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
# ]
# ///

"""ボイスクローン TTS サンプルクライアント。

m4a 等のリファレンス音声ファイルを使い、任意のテキストを
その声で読み上げる音声を生成します。

前提:
  - サーバーが同じホストで起動していること (./run_mlx_server.sh)
  - ref_audio に指定するパスは「サーバーから見たパス」であること
    (サーバーと同じMacで動かす前提なので、ローカルパスでOK)

使い方:
  # リファレンス音声の書き起こしテキストは必須（クローン精度に直結）
  uv run scripts/tts_voice_clone.py \
      --ref-audio /path/to/voice.m4a \
      --ref-text "リファレンス音声に含まれる発話内容" \
      --input "こんにちは。この声で読み上げます。" \
      --output speech.mp3

  # 入力テキストをファイルから読ませる
  uv run scripts/tts_voice_clone.py \
      --ref-audio /path/to/voice.m4a \
      --ref-text "..." \
      --input-file manuscript.txt \
      --output speech.wav

  # 1.7B モデル（より安定、やや遅い）
  uv run scripts/tts_voice_clone.py \
      --ref-audio /path/to/voice.m4a \
      --ref-text "..." \
      --input "..." \
      --model qwen3-tts-1.7b-base-8bit

  # Irodori（日本語特化）: 書き起こし不要
  uv run scripts/tts_voice_clone.py \
      --ref-audio /path/to/voice.m4a \
      --input "こんにちは。この声で読み上げます。" \
      --model irodori-tts-500m-v3-8bit \
      --output speech.wav

  # Irodori VoiceDesign: 声質を言葉で指示（リファレンス音声なしでも可）
  uv run scripts/tts_voice_clone.py \
      --instruct "落ち着いた女性の声で、やわらかく自然に読み上げてください。" \
      --input "こんにちは。" \
      --model irodori-tts-600m-v3-voicedesign-8bit \
      --output designed.wav

エンジンによる違い:
  - Qwen3-TTS: --ref-audio と --ref-text（書き起こし）が必須。--lang / --max-tokens が使える
  - Irodori:   --ref-audio だけでクローンできる。--ref-text は無視される。
               長さは --seconds / --duration-scale で制御する（--speed も内部で変換される）
"""

import argparse
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:1235"
DEFAULT_MODEL = "qwen3-tts-0.6b-base-8bit"
SUPPORTED_FORMATS = ("mp3", "wav", "flac", "ogg")

QWEN3_MODELS = ("qwen3-tts-0.6b-base-8bit", "qwen3-tts-1.7b-base-8bit")
IRODORI_BASE_MODELS = (
    "irodori-tts-500m-v3-fp16",
    "irodori-tts-500m-v3-8bit",
    "irodori-tts-500m-v2-fp16",
    "irodori-tts-500m-v2-8bit",
)
IRODORI_VOICE_DESIGN_MODELS = (
    "irodori-tts-600m-v3-voicedesign-fp16",
    "irodori-tts-600m-v3-voicedesign-8bit",
)
IRODORI_MODELS = IRODORI_BASE_MODELS + IRODORI_VOICE_DESIGN_MODELS
ALL_MODELS = QWEN3_MODELS + IRODORI_MODELS
# duration predictor 非搭載。seconds 未指定だと 30 秒固定 (約 24GB) になる
IRODORI_NO_DURATION_PREDICTOR = ("irodori-tts-500m-v2-fp16", "irodori-tts-500m-v2-8bit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ボイスクローン TTS: m4a リファレンス音声から声をクローンして音声合成",
    )
    p.add_argument(
        "--ref-audio",
        default=None,
        help="リファレンス音声ファイルのパス（サーバーから見たパス）。5〜15秒推奨。"
             "Qwen3-TTS では必須、Irodori VoiceDesign では --instruct があれば省略可",
    )
    p.add_argument(
        "--ref-text",
        default=None,
        help="リファレンス音声の書き起こしテキスト。ファイルパスの場合はファイル内容を読み込む。"
             "Qwen3-TTS の ICL では必須（クローン精度に直結）。Irodori では不要で、指定しても無視される",
    )
    p.add_argument(
        "--instruct",
        default=None,
        help="声質を言葉で指示する（Irodori VoiceDesign 版のみ）。例: 落ち着いた女性の声で、やわらかく",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="読み上げるテキスト")
    src.add_argument("--input-file", help="読み上げるテキストが入ったファイルパス")
    p.add_argument(
        "--output",
        default="speech.mp3",
        help="出力ファイル名（デフォルト: speech.mp3）",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=ALL_MODELS,
        help=f"TTSモデル（デフォルト: {DEFAULT_MODEL}）。irodori-* は日本語特化エンジン",
    )
    p.add_argument(
        "--format",
        default=None,
        choices=SUPPORTED_FORMATS,
        help="出力フォーマット。未指定時は --output の拡張子から自動判定",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="再生速度（デフォルト: 1.0）",
    )
    p.add_argument(
        "--lang",
        default="auto",
        choices=("auto", "japanese", "chinese", "english", "korean",
                 "german", "french", "italian", "portuguese", "spanish", "russian"),
        help="言語コード（デフォルト: auto）。日本語の場合は japanese を指定",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大トークン数。--lang 指定時のボイスクローンでは自動で 8192",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="出力長を秒で明示（Irodori のみ）。v2 では未指定だと 30 秒固定・約24GB になるため実質必須",
    )
    p.add_argument(
        "--duration-scale",
        type=float,
        default=None,
        help="推定長に対する倍率（Irodori v3 のみ、>1 で長く）。--speed より優先される",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Euler ステップ数（Irodori のみ、既定 40）。6 程度まで下げると高速化する",
    )
    p.add_argument(
        "--cfg-guidance-mode",
        default=None,
        choices=("independent", "alternating"),
        help="CFG のガイダンス方式（Irodori のみ）。alternating はメモリが約 1/3 になる",
    )
    p.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"サーバーURL（デフォルト: {BASE_URL}）",
    )
    return p.parse_args()


def resolve_format(output_path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    ext = Path(output_path).suffix.lstrip(".").lower()
    if ext in SUPPORTED_FORMATS:
        return ext
    print(
        f"警告: 出力拡張子 '.{ext}' は未対応です。mp3 を使用します。",
        file=sys.stderr,
    )
    return "mp3"


def read_input_text(args: argparse.Namespace) -> str:
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8").strip()
    return args.input.strip()


def main() -> None:
    args = parse_args()

    is_irodori = args.model in IRODORI_MODELS
    is_voice_design = args.model in IRODORI_VOICE_DESIGN_MODELS

    ref_audio = None
    if args.ref_audio:
        ref_audio = Path(args.ref_audio).resolve()
        if not ref_audio.exists():
            print(f"エラー: リファレンス音声が見つかりません: {ref_audio}", file=sys.stderr)
            sys.exit(1)

    if not is_irodori and not ref_audio:
        print("エラー: Qwen3-TTS のボイスクローンには --ref-audio が必要です。", file=sys.stderr)
        sys.exit(1)
    if is_voice_design and not (ref_audio or args.instruct):
        print("エラー: VoiceDesign 版には --instruct か --ref-audio が必要です。", file=sys.stderr)
        sys.exit(1)
    if is_irodori and not is_voice_design and args.instruct:
        print(
            f"エラー: {args.model} は caption 条件を持たないため --instruct を使えません。"
            " VoiceDesign 版を指定してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    if not is_irodori and not args.ref_text:
        print("エラー: Qwen3-TTS の ICL には --ref-text（書き起こし）が必要です。", file=sys.stderr)
        sys.exit(1)

    text = read_input_text(args)
    if not text:
        print("エラー: 読み上げテキストが空です。", file=sys.stderr)
        sys.exit(1)

    response_format = resolve_format(args.output, args.format)

    payload = {
        "input": text,
        "model": args.model,
        "response_format": response_format,
    }
    if ref_audio:
        payload["ref_audio"] = str(ref_audio)
    if args.speed != 1.0:
        payload["speed"] = args.speed

    if is_irodori:
        # Irodori はリファレンス音声だけでクローンでき、書き起こしは使わない
        if args.ref_text:
            print("注意: Irodori は書き起こしを使わないため --ref-text は無視されます。", file=sys.stderr)
        if args.instruct:
            payload["instruct"] = args.instruct
        if args.seconds is not None:
            payload["seconds"] = args.seconds
        elif args.model in IRODORI_NO_DURATION_PREDICTOR:
            print(
                f"注意: {args.model} は duration predictor 非搭載です。--seconds 未指定だと"
                " 30 秒固定（約24GB）で生成されます。",
                file=sys.stderr,
            )
        if args.duration_scale is not None:
            payload["duration_scale"] = args.duration_scale
        if args.num_steps:
            payload["num_steps"] = args.num_steps
        if args.cfg_guidance_mode:
            payload["cfg_guidance_mode"] = args.cfg_guidance_mode
    else:
        # ref_text がファイルパスならファイル内容を読み込む
        ref_text = args.ref_text
        ref_text_path = Path(ref_text)
        if ref_text_path.exists() and ref_text_path.is_file():
            ref_text = ref_text_path.read_text(encoding="utf-8").strip()
        payload["ref_text"] = ref_text
        if args.lang != "auto":
            payload["lang_code"] = args.lang
        if args.max_tokens:
            payload["max_tokens"] = args.max_tokens

    print(f"モデル:        {args.model} ({'Irodori' if is_irodori else 'Qwen3-TTS'})")
    print(f"リファレンス:  {ref_audio or '(なし)'}")
    if is_voice_design and args.instruct:
        print(f"声質の指示:    {args.instruct}")
    print(f"フォーマット:  {response_format}")
    if not is_irodori:
        print(f"言語:          {args.lang}")
    print(f"テキスト長:    {len(text)} 文字")
    print("リクエスト送信中...")

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{args.base_url}/v1/audio/speech", json=payload)
    except httpx.ConnectError:
        print(
            f"エラー: サーバーに接続できません。起動していますか？ {args.base_url}",
            file=sys.stderr,
        )
        sys.exit(1)

    if resp.status_code != 200:
        print(
            f"エラー: サーバーが {resp.status_code} を返しました: {resp.text[:500]}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(args.output)
    output_path.write_bytes(resp.content)
    print(f"完了: {output_path} ({len(resp.content):,} bytes)")


if __name__ == "__main__":
    main()
