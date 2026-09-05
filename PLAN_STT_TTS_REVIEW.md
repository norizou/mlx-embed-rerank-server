# Opus技術レビュー: Qwen3 STT/TTS エンドポイント実装プラン

> **【アーカイブ】これは実装前の設計レビュー記録です（2026-06-25）。**
> 正本は `mlx_embed_rerank_server.py` の実装と `README.md` / `README_JA.md` です。
> 実装時に本プランから変更された主な点:
>
> | 項目 | 本プランの記載 | 実装（現行コード） |
> |:---|:---|:---|
> | ASR デフォルト | `qwen3-asr-0.6b-8bit` | **`qwen3-asr-1.7b-8bit`**（精度優先。速度差は +0.25 秒） |
> | TTS デフォルト | `qwen3-tts-0.6b-base-4bit` | **`qwen3-tts-0.6b-base-8bit`**（4bit は削除し 8bit に統一） |
> | 登録モデル数 | ASR 1 / TTS 1 | ASR 2（0.6B/1.7B 8bit）/ TTS 2（0.6B/1.7B 8bit） |
> | STT の音声前処理 | `audio_read(BytesIO)` → `audio_write(tmp)` | アップロードされた**バイト列を拡張子ごと `/tmp` に直接書き出し**てパスを渡す（m4a/mp4 の `moov` 末尾配置でパイプ入力がシークできず ffmpeg が空配列を返すため） |
> | モデル定義 | `loader` キーを持つ | `loader` キーは持たず `type`（`asr` / `tts`）で判別 |
> | 出力フォーマット | wav / mp3 / flac | wav / mp3 / flac / **ogg** |
> | ヘルスチェック | 記載なし | ポート **1236** の専用サーバーを追加（`BENCHMARK_REPORT.md` §6.6） |
>
> 変更の根拠となる実測値は `BENCHMARK_REPORT.md` の第5章・第6章を参照してください。


## レビュー概要

Kimiが設計した実装プランについて、HuggingFaceモデルカード・`mlx-audio`ライブラリのソースコード・既存の`mlx-audio`サーバー実装を精査した結果、**重大な修正が必要な箇所が複数発見された**。以下に修正版プランを提示する。

---

## 1. 致命的な誤り: モデル読み込みライブラリ

### Kimiプランの問題点

> `mlx-lm.load()` を第一候補とし、必要に応じて `AutoProcessor`/`AutoTokenizer` を併用

**これは完全に誤り。** 両モデルとも `mlx-audio` ライブラリ専用のモデルであり、`mlx-lm` や `mlx-embeddings` では読み込めない。

### 正しい読み込み方法

| 機能 | ライブラリ | 読み込み関数 |
|:---|:---|:---|
| STT | `mlx-audio` | `from mlx_audio.stt.utils import load_model` |
| TTS | `mlx-audio` | `from mlx_audio.tts.utils import load_model` |

**STT (Qwen3-ASR-0.6B-8bit):**
```python
from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription

model = load_model("mlx-community/Qwen3-ASR-0.6B-8bit")
result = generate_transcription(
    model=model,
    audio="path_to_audio.wav",  # ファイルパス or mx.array
    output_path="/dev/null",
    format="txt",
    verbose=False,
)
print(result.text)  # 認識結果テキスト
```

**TTS (Qwen3-TTS-12Hz-0.6B-Base-4bit):**
```python
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")

# model.generate() はジェネレータ（各resultに .audio と .sample_rate がある）
for result in model.generate("こんにちは", voice="Chelsie", lang_code="ja"):
    audio_array = result.audio      # numpy/mx.array 波形
    sample_rate = result.sample_rate # サンプリングレート
```

> **重要**: `generate_audio()` はCLI用ヘルパーで `None` を返す（ファイルに直接書き出す）。API統合では `model.generate()` を直接使用する。

---

## 2. 依存関係の修正

### Kimiプラン
```
python-multipart, soundfile
```

### 修正版

| パッケージ | 必要性 | 理由 |
|:---|:---|:---|
| `mlx-audio` | **必須（新規追加）** | STT/TTSの全機能を提供。`load_model`, `generate`, `audio_io` 含む |
| `python-multipart` | **必須（新規追加）** | FastAPIのファイルアップロード（`UploadFile`, `File`, `Form`）に必要 |
| ~~`soundfile`~~ | **不要** | `mlx-audio` が内蔵の `mlx_audio.audio_io.read/write` で音声I/Oを処理。BytesIOバッファへの書き込みも対応 |

**pyproject.toml 修正案:**
```toml
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "mlx",
    "mlx-embeddings",
    "mlx-lm",
    "mlx-audio",           # 追加: STT/TTS機能
    "python-multipart",    # 追加: ファイルアップロード
    "numpy",
]
```

### ffmpeg について

- **システム要件として必要**（`mlx_audio.audio_io.read` が内部で ffmpeg を呼び出し、mp4/m4a/webm等から音声を抽出する）
- ただし **Python依存関係としては不要**（サブプロセスで呼び出される）
- README に `brew install ffmpeg` を記載する

---

## 3. 音声前処理の修正

### Kimiプランの問題点

> `normalize_audio()` を自作し、`soundfile` や `ffmpeg` で手動変換する

**不要。** `mlx-audio` が全て処理する。

### 正しいアプローチ

**STT入力:**
```python
from mlx_audio.audio_io import read as audio_read

# BytesIOからの読み込み（アップロードされたバイナリ）
data = await file.read()
audio_array, sample_rate = audio_read(io.BytesIO(data), always_2d=False)

# 一時ファイルに書き出してモデルに渡す（mlx-audio serverと同じ方式）
from mlx_audio.audio_io import write as audio_write
tmp_path = f"/tmp/{uuid.uuid4()}.wav"
audio_write(tmp_path, audio_array, sample_rate)

# モデルに一時ファイルのパスを渡す
result = model.generate(tmp_path, **kwargs)
```

**TTS出力:**
```python
from mlx_audio.audio_io import write as audio_write

# BytesIOバッファに直接書き込み
buffer = io.BytesIO()
audio_write(buffer, concatenated_audio, sample_rate, format="mp3")
buffer.seek(0)
return StreamingResponse(buffer, media_type="audio/mpeg")
```

**対応フォーマット** (mlx_audio.audio_io経由):
- 入力: wav, mp3, flac, m4a, mp4, webm, ogg, aac, opus（ffmpeg経由）
- 出力: wav, mp3, flac, ogg（audio_write経由）

---

## 4. STT エンドポイント修正版

### リクエスト仕様 (変更なし)
```http
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|:---|:---|:---|:---|:---|
| `file` | File | Yes | - | 音声ファイル（mp4, wav, mp3等） |
| `model` | string | No | `qwen3-asr-0.6b-8bit` | モデルID |
| `language` | string | No | `None` | 言語コード（例: `ja`, `en`） |

### レスポンス仕様 (OpenAI互換)
```json
{
  "text": "認識されたテキストです。"
}
```

### 実装フロー（修正版）

```python
from fastapi import File, Form, UploadFile
from mlx_audio.audio_io import read as audio_read, write as audio_write

DEFAULT_ASR = "qwen3-asr-0.6b-8bit"

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_ASR),
    language: Optional[str] = Form(None),
):
    # 1. アップロードされた音声を読み込み
    data = await file.read()
    audio_array, sr = audio_read(io.BytesIO(data), always_2d=False)

    # 2. 一時ファイルに書き出し（mlx-audio STTモデルはファイルパスを期待）
    tmp_path = f"/tmp/stt_{uuid.uuid4()}.wav"
    audio_write(tmp_path, audio_array, sr)

    try:
        # 3. モデルをlazy load
        stt_model = manager.get_asr(model)

        # 4. 推論実行
        kwargs = {}
        if language:
            kwargs["language"] = language
        result = stt_model.generate(tmp_path, **kwargs)

        # 5. resultから.textを取得して返却
        text = result.text if hasattr(result, "text") else str(result)
        return {"text": text}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
```

---

## 5. TTS エンドポイント修正版

### リクエスト仕様
```http
POST /v1/audio/speech
Content-Type: application/json
```

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|:---|:---|:---|:---|:---|
| `input` | string | Yes | - | 読み上げるテキスト |
| `model` | string | No | `qwen3-tts-0.6b-base-4bit` | モデルID |
| `voice` | string | No | `None` | 話者名（例: `Chelsie`, `Ethan`） |
| `response_format` | string | No | `mp3` | 出力形式: wav, mp3, flac |
| `speed` | float | No | `1.0` | 再生速度 |
| `ref_audio` | string | No | `None` | ボイスクローン用リファレンス音声パス |
| `ref_text` | string | No | `None` | リファレンス音声のテキスト |

> **Kimiプランにない重要な追加**: `voice`, `ref_audio`, `ref_text` パラメータ。Qwen3-TTSの主要機能であるボイスクローンをサポートする。

### レスポンス仕様
- 音声バイナリを直接返却
- `Content-Type`: `audio/mpeg`（mp3）、`audio/wav`（wav）、`audio/flac`（flac）

### 実装フロー（修正版）

```python
import io
import numpy as np
from fastapi.responses import StreamingResponse
from mlx_audio.audio_io import write as audio_write

class SpeechReq(BaseModel):
    input: str
    model: Optional[str] = DEFAULT_TTS
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None

@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechReq):
    model_name = req.model or DEFAULT_TTS
    tts_model = manager.get_tts(model_name)

    # model.generate() はジェネレータ
    generate_kwargs = {}
    if req.voice:
        generate_kwargs["voice"] = req.voice
    if req.speed and req.speed != 1.0:
        generate_kwargs["speed"] = req.speed
    if req.ref_audio:
        generate_kwargs["ref_audio"] = req.ref_audio
    if req.ref_text:
        generate_kwargs["ref_text"] = req.ref_text

    # 全チャンクを収集
    audio_chunks = []
    sample_rate = None
    for result in tts_model.generate(req.input, **generate_kwargs):
        audio_chunks.append(result.audio)
        if sample_rate is None:
            sample_rate = result.sample_rate

    if not audio_chunks:
        raise HTTPException(status_code=400, detail="No audio generated")

    # 結合してエンコード
    concatenated = np.concatenate(audio_chunks)
    buffer = io.BytesIO()
    audio_write(buffer, concatenated, sample_rate, format=req.response_format)
    buffer.seek(0)

    content_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }

    return StreamingResponse(
        buffer,
        media_type=content_type_map.get(req.response_format, "audio/mpeg"),
        headers={
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}"
        },
    )
```

---

## 6. ModelManager 拡張（修正版）

### Kimiプランとの差分

Kimiプランでは `mlx-lm` の `lm_load` でモデルを読み込む前提だったが、正しくは `mlx_audio` 専用のローダーを使う。

```python
# imports 追加
try:
    from mlx_audio.stt.utils import load_model as stt_load_model
except ImportError:
    print("Warning: mlx-audio STT not found.")
    stt_load_model = None

try:
    from mlx_audio.tts.utils import load_model as tts_load_model
except ImportError:
    print("Warning: mlx-audio TTS not found.")
    tts_load_model = None

# モデル定義
AVAILABLE_AUDIO_MODELS = {
    "qwen3-asr-0.6b-8bit": {
        "id": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "type": "asr",
        "loader": "stt",
        "description": "Qwen3 ASR 0.6B 8-bit (Speech-to-Text)"
    },
    "qwen3-tts-0.6b-base-4bit": {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
        "type": "tts",
        "loader": "tts",
        "description": "Qwen3 TTS 0.6B Base 4-bit (Text-to-Speech, Voice Clone)"
    },
}

DEFAULT_ASR = "qwen3-asr-0.6b-8bit"
DEFAULT_TTS = "qwen3-tts-0.6b-base-4bit"
```

### ModelManager に追加するメソッド

```python
class ModelManager:
    def __init__(self, ...):
        ...
        self.asr_cache = {}   # 追加
        self.tts_cache = {}   # 追加

    def get_asr(self, name: str):
        if name not in AVAILABLE_AUDIO_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported ASR model: {name}")
        config = AVAILABLE_AUDIO_MODELS[name]
        if config["type"] != "asr":
            raise HTTPException(status_code=400, detail=f"Model {name} is not an ASR model")
        with self.lock:
            if name not in self.asr_cache:
                print(f"Loading ASR model: {config['id']}...")
                self.asr_cache[name] = stt_load_model(config['id'])
                self.last_used[name] = time.time()
            self.last_used[name] = time.time()
            return self.asr_cache[name]

    def get_tts(self, name: str):
        if name not in AVAILABLE_AUDIO_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported TTS model: {name}")
        config = AVAILABLE_AUDIO_MODELS[name]
        if config["type"] != "tts":
            raise HTTPException(status_code=400, detail=f"Model {name} is not a TTS model")
        with self.lock:
            if name not in self.tts_cache:
                print(f"Loading TTS model: {config['id']}...")
                self.tts_cache[name] = tts_load_model(config['id'])
                self.last_used[name] = time.time()
            self.last_used[name] = time.time()
            return self.tts_cache[name]
```

---

## 7. ヘルスチェック拡張

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_embed_models": list(manager.embed_cache.keys()),
        "loaded_rerank_models": list(manager.rerank_cache.keys()),
        "loaded_asr_models": list(manager.asr_cache.keys()),    # 追加
        "loaded_tts_models": list(manager.tts_cache.keys()),    # 追加
        "available_embed": list(AVAILABLE_EMBED_MODELS.keys()),
        "available_rerank": list(AVAILABLE_RERANK_MODELS.keys()),
        "available_audio": list(AVAILABLE_AUDIO_MODELS.keys()), # 追加
    }
```

---

## 8. Kimiプランとの差分まとめ

| 項目 | Kimiプラン | Opus修正 | 理由 |
|:---|:---|:---|:---|
| **モデル読み込み** | `mlx-lm.load()` / `AutoProcessor` | `mlx_audio.stt/tts.utils.load_model` | 専用ライブラリが必要 |
| **依存関係** | `soundfile` 追加 | `mlx-audio` 追加、`soundfile` 削除 | `mlx-audio` が音声I/Oを内蔵 |
| **音声前処理** | 自作 `normalize_audio()` | `mlx_audio.audio_io.read/write` | 車輪の再発明を回避 |
| **TTS推論** | `mlx-lm` でテキスト生成→音声化 | `model.generate()` ジェネレータ | TTSモデルは直接音声波形を生成 |
| **出力エンコード** | `ffmpeg` サブプロセス | `mlx_audio.audio_io.write` | BytesIOへの直接書き込み対応 |
| **ボイスクローン** | 未記載 | `ref_audio`, `ref_text` パラメータ追加 | Qwen3-TTS Base の主要機能 |
| **voice パラメータ** | 未記載 | `voice` パラメータ追加 | 話者選択機能 |
| **ffmpeg依存** | Python依存として追加 | システム要件として記載のみ | サブプロセス呼び出しのため |

---

## 9. 修正対象ファイル一覧

| ファイル | 変更内容 |
|:---|:---|
| `pyproject.toml` | `mlx-audio`, `python-multipart` を追加 |
| `mlx_embed_rerank_server.py` | `mlx-audio` import追加、`AVAILABLE_AUDIO_MODELS`定義、`ModelManager`にasr/ttsキャッシュ追加、`/v1/audio/transcriptions`・`/v1/audio/speech`エンドポイント追加、`/health`拡張 |
| `tests/test_api.py` | STT/TTSエンドポイントのテスト追加 |
| `README.md` | 新エンドポイント・モデル・ffmpeg要件のドキュメント追加 |
| `README_JA.md` | 日本語版同期 |

---

## 10. 検証コマンド

```bash
# ffmpeg確認
ffmpeg -version

# サーバー起動
./run_mlx_server.sh

# ヘルスチェック
curl http://localhost:1235/health

# STT（wav）
curl -X POST http://localhost:1235/v1/audio/transcriptions \
  -F file="@sample.wav" \
  -F model="qwen3-asr-0.6b-8bit"

# STT（mp4）
curl -X POST http://localhost:1235/v1/audio/transcriptions \
  -F file="@sample.mp4" \
  -F model="qwen3-asr-0.6b-8bit" \
  -F language="ja"

# TTS（mp3出力）
curl -X POST http://localhost:1235/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "こんにちは、これはテストです。",
    "model": "qwen3-tts-0.6b-base-4bit",
    "voice": "Chelsie",
    "response_format": "mp3"
  }' \
  --output output.mp3

# TTS（ボイスクローン）
curl -X POST http://localhost:1235/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "この声をクローンします。",
    "model": "qwen3-tts-0.6b-base-4bit",
    "ref_audio": "/path/to/reference.wav",
    "ref_text": "リファレンス音声のテキスト",
    "response_format": "wav"
  }' \
  --output cloned.wav
```

---

## 11. リスク・考慮事項（修正版）

| 項目 | 詳細 | 対応方針 |
|:---|:---|:---|
| **mlx-audio バージョン** | `mlx-audio>=0.3.0` が必要（Qwen3-TTS対応）。最新 v0.4.3 推奨 | `pyproject.toml` にバージョン制約を記載 |
| **ffmpeg システム依存** | mp4/m4a/webm の音声読み込みにシステム ffmpeg が必要 | README に `brew install ffmpeg` を明記 |
| **メモリ使用量** | ASR 0.4B + TTS 0.4B + 既存モデルの同時ロード | 0.6B級は軽量。初期は inactivity unload 不要。必要なら後で追加 |
| **TTS ボイスクローン** | `ref_audio` はサーバー上のファイルパスが必要。クライアントからのアップロードは別途対応検討 | Phase 1 はサーバーローカルパスのみ。Phase 2 でアップロード対応 |
| **mlx-audio の依存関係の重さ** | `mlx-audio` は `webrtcvad`, `sounddevice` 等の追加依存を持つ可能性 | 必要最小限のサブモジュール import で対応。問題があれば optional dependency 化 |
| **STT model.generate() の戻り値** | Qwen3-ASR の `generate()` が Whisper 型 output（`.text`, `.segments`）を返すか要実機確認 | 実装時に `hasattr` で分岐し、`.text` を優先取得 |
| **TTS の日本語品質** | Qwen3-TTS は10言語対応だが、日本語の自然さはモデル依存 | 実装後に品質確認し、必要に応じてREADMEに記載 |

---

## 12. 実装ステップ（修正版・優先順位順）

1. **`pyproject.toml` に `mlx-audio`, `python-multipart` を追加** → `uv sync`
2. **`mlx_embed_rerank_server.py`** に以下を追加:
   - `mlx_audio` の import（`try/except` で graceful fallback）
   - `AVAILABLE_AUDIO_MODELS`, `DEFAULT_ASR`, `DEFAULT_TTS` 定義
   - `ModelManager` に `asr_cache`, `tts_cache`, `get_asr()`, `get_tts()` 追加
   - `POST /v1/audio/transcriptions` エンドポイント
   - `POST /v1/audio/speech` エンドポイント
   - `/health` の拡張
3. **動作確認**: サーバー起動 → curl で STT/TTS テスト
4. **テスト追加**: `tests/test_api.py`
5. **ドキュメント更新**: `README.md`, `README_JA.md`
