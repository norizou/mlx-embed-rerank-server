# MLX Embedding & Reranker Server

[English](README.md) | 日本語

Apple Silicon 向けにネイティブ最適化された **MLX** バックエンドで、
**Embedding / Rerank / STT（音声認識） / TTS（音声合成）** を
**1 つの FastAPI プロセス**で提供する軽量 API サーバーです。

LM Studio や Ollama などの LLM サーバーとは独立して常駐させ、
RAG（Retrieval-Augmented Generation）用の **Embedding / Rerank / 音声エンジン**として利用することを想定しています。

### 🚀 デフォルトモデル
`model` パラメータを省略した場合に使われるモデルです。選定根拠は [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) の実測結果に基づきます。

- **Embedding**: `bge-m3`（*bge-m3-mlx-fp16*）
- **Reranker**: `qwen3-0.6b`（*Qwen3-Reranker-0.6B-mxfp8*）
- **ASR（STT）**: `qwen3-asr-1.7b-8bit`（*Qwen3-ASR-1.7B-8bit*）
- **TTS**: `qwen3-tts-0.6b-base-8bit`（*Qwen3-TTS-12Hz-0.6B-Base-8bit*）

モデルは**起動時ではなく初回リクエスト時に遅延ロード**されます。重い Qwen3-VL 系モデルが無使用でアンロードされた後は、デフォルトの Embedding / Rerank モデルが自動でプリロードされ、ウォーム状態で待機します。

---

## ✨ 特徴

- ✅ Embedding / Rerank / 音声を **1 プロセス**で提供（API ポート `1235`、ヘルス専用ポート `1236`）
- ✅ OpenAI 互換 API（`/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech`）
- ✅ Apple Silicon ネイティブの **MLX** による GPU 推論
- ✅ Embedding 5 モデル / Reranker 2 モデル / 音声 4 モデルをリクエスト単位で切り替え
- ✅ Qwen3-VL Embedding / Reranker 2B（`instruction` 対応。`torch` / `torchvision` が別途必要 → [動作環境](#-動作環境)）
- ✅ STT（`/v1/audio/transcriptions`）と、ボイスクローン対応 TTS（`/v1/audio/speech`）
- ✅ Qwen3-VL の自動アンロード + デフォルトモデルのプリロード（メモリ最適化）
- ✅ 推論中でも応答するヘルスチェック専用サーバー（別スレッド / 別ポート）
- ✅ GGUF 変換不要。MLX コミュニティの重みをそのまま利用
- ✅ OpenWebUI / Dify / LangChain / 自作 RAG パイプラインに接続可能

---

## 🧠 提供 API

### ベース URL

```
http://localhost:1235
```

ヘルスチェック専用サーバーが `http://localhost:1236` でも待ち受けます（→ [スーパーバイザー監視機能](#️-スーパーバイザー監視機能自動再起動)）。

### エンドポイント一覧

| Method | Path                    | 内容 |
|------|--------------------------|------|
| GET  | `/health`               | ロード済み／利用可能モデルを含むヘルスチェック |
| POST | `/v1/embeddings`        | **テキスト**の埋め込み生成（OpenAI 互換） |
| POST | `/v1/rerank`（別名 `/rerank`） | クエリと文書群のリランク |
| POST | `/v1/audio/transcriptions` | 音声ファイルの文字起こし（STT、OpenAI 互換） |
| POST | `/v1/audio/speech`      | テキストからの音声合成（TTS、OpenAI 互換） |

> **注意**: 画像入力は HTTP API では公開していません。Qwen3-VL 系モデルもテキスト専用の Embedding / Rerank として提供されます（`input` は文字列または文字列配列のみ）。

### `GET /health`

```json
{
  "status": "ok",
  "loaded_embed_models": [], "loaded_rerank_models": [],
  "loaded_asr_models": [], "loaded_tts_models": [],
  "available_embed": ["gemma-3-300m", "bge-m3", "bge-m3-8bit", "qwen3-vl-embedding-2b", "qwen3-0.6b-embed"],
  "available_rerank": ["qwen3-0.6b", "qwen3-vl-reranker-2b"],
  "available_audio": ["qwen3-asr-0.6b-8bit", "qwen3-asr-1.7b-8bit", "qwen3-tts-0.6b-base-8bit", "qwen3-tts-1.7b-base-8bit"]
}
```

### `POST /v1/embeddings`

| フィールド | 型 | デフォルト | 説明 |
|:---|:---|:---|:---|
| `input` | `string \| string[]` | 必須 | 埋め込み対象のテキスト |
| `model` | `string` | `bge-m3` | Embedding モデル ID |
| `input_type` | `"query" \| "document"` | `"document"` | `gemma-3-300m` の自動プレフィックス、`qwen3-vl-embedding-2b` の既定 instruction を切り替え |
| `instruction` | `string` | `null` | **`qwen3-vl-embedding-2b` 専用**。他の Embedding モデルでは無視されます |

**出力ベクトルの仕様**: すべての Embedding モデルで `last_hidden_state` の先頭トークン（CLS プーリング）を取り出し、L2 正規化した単位ベクトルを返します。これは BGE-M3 の dense embedding の仕様に一致しており、`bge-m3` をデフォルトかつ日本語 RAG の推奨としている理由です（→ [モデル選定の指針](#-モデル選定の指針)）。

### `POST /v1/rerank`（別名 `POST /rerank`）

| フィールド | 型 | デフォルト | 説明 |
|:---|:---|:---|:---|
| `query` | `string` | 必須 | 検索クエリ |
| `documents` | `string[]` | 必須 | 候補文書 |
| `model` | `string` | `qwen3-0.6b` | Reranker モデル ID |
| `top_k` | `int` | `10` | 返却件数。`0` を指定すると全件返却 |
| `instruction` | `string` | `null` | **`qwen3-vl-reranker-2b` 専用**。`qwen3-0.6b` は固定の Yes/No 判定プロンプトを使うため無視されます |

レスポンスは `{"model": ..., "results": [{"index": ..., "relevance_score": ...}, ...]}` でスコア降順です。`qwen3-0.6b` のスコアは `yes` / `no` トークンのロジットに対する softmax 確率のため絶対値は小さくなります。**順位**が信号であり、スコアの絶対値ではありません。

### エラー応答の挙動

| 状況 | エンドポイント | ステータス |
|:---|:---|:---|
| 未対応のモデル名 | `/v1/rerank`, `/v1/audio/*` | `400`（短い detail メッセージ） |
| 未対応のモデル名 | `/v1/embeddings` | `500`（`compute_embeddings` を包む catch-all ハンドラが内部の 400 ごと 500 に変換するため） |
| 推論の失敗 | `/v1/embeddings` | `500` + `detail` に Python トレースバック全文（ローカルデバッグ用途。ポートを外部公開しないこと） |
| 推論の失敗 | `/v1/rerank` | `500 Internal Server Error`（トレースバックなし。サーバーログを参照） |

---

## ⏱️ 自動フォールバック（Auto Fallback）

Mac のユニファイドメモリを節約するため、重い Qwen3-VL 系モデル（`qwen3-vl-embedding-2b` / `qwen3-vl-reranker-2b`）は **30 秒間未使用でアンロード**されます。

- **タイマー粒度**: 判定タイマーも 30 秒周期のため、実際の解放は最終リクエストから 30〜60 秒後になります。
- **ペアアンロード**: どちらか一方がタイムアウトしたら、Qwen3-VL の embed / rerank を**両方まとめて解放**します。
- **デフォルトのプリロード**: 同時に軽量なデフォルトモデル（`bge-m3` / `qwen3-0.6b`）をロードし、通常のリクエストに即応できる状態にします。
- **GPU メモリの即時解放**: アンロードとプリロードを終えた最後に `mx.metal.clear_cache()` を実行します。
- **音声モデルは対象外**: ASR / TTS モデルは一度ロードすると常駐します（0.6B / 1.7B 級は軽量なためアンロード不要と判断）。

### モデル状態遷移図

```mermaid
stateDiagram-v2
    [*] --> Empty : サーバー起動（モデル未ロード）
    Empty --> DefaultLoaded : 初回リクエスト（bge-m3 / qwen3-0.6b を遅延ロード）
    Empty --> Qwen3VL_Loaded : 初回リクエストで qwen3-vl 指定（遅延ロード）
    DefaultLoaded --> Qwen3VL_Loaded : /v1/embeddings または /v1/rerank で qwen3-vl 指定
    Qwen3VL_Loaded --> Qwen3VL_Loaded : 30秒以内にリクエスト（last_used 更新）
    Qwen3VL_Loaded --> Fallback : 30秒無使用（30秒周期タイマーで判定＝実際は30〜60秒）
    Fallback --> DefaultLoaded : デフォルトモデルをプリロードして待機
    Fallback --> Qwen3VL_Loaded : qwen3-vl 指定のリクエスト受信

    state Qwen3VL_Loaded {
        [*] --> Embed_Rerank_Loaded
        Embed_Rerank_Loaded --> [*]
    }

    state Fallback {
        [*] --> Unload_Qwen3VL : ペアアンロード
        Unload_Qwen3VL --> Load_Default : bge-m3 / qwen3-0.6b をプリロード
        Load_Default --> [*] : mx.metal.clear_cache()
    }
```

### フォールバックシーケンス図

```mermaid
sequenceDiagram
    autonumber
    participant Client as API クライアント
    participant API as FastAPI (/v1/embeddings)
    participant MM as ModelManager
    participant Timer as FallbackTimer (30秒周期)
    participant GPU as Apple Metal GPU

    Client->>API: POST /v1/embeddings (qwen3-vl-embedding-2b)
    API->>MM: get_embed("qwen3-vl-embedding-2b")
    MM->>MM: emb_load() + _fix_qwen3vl_processor()
    MM-->>API: (model, processor)
    API-->>Client: embedding output

    Client->>API: POST /v1/rerank (qwen3-vl-reranker-2b)
    API->>MM: get_rerank("qwen3-vl-reranker-2b")
    MM->>MM: emb_load() + _fix_qwen3vl_processor()
    MM-->>API: (model, processor)
    API-->>Client: rerank output

    Note over Client,GPU: 30秒間リクエストなし

    Timer->>Timer: _check_inactivity()
    Timer->>MM: qwen3vl_embed_timedout = True
    Timer->>MM: qwen3vl_rerank_timedout = True

    MM->>MM: [Fallback] Unload embed 'qwen3-vl-embedding-2b' (paired unload)
    MM->>MM: [Fallback] Unload rerank 'qwen3-vl-reranker-2b' (paired unload)
    MM->>MM: [Fallback] Preload default embed 'bge-m3'
    MM->>MM: [Fallback] Preload default rerank 'qwen3-0.6b'
    MM->>GPU: mx.metal.clear_cache()
```

### メモリ状態の比較

| 状態 | ロード済み Embed | ロード済み Rerank | メモリ使用量 | 次のリクエスト |
|------|-----------------|------------------|-------------|--------------|
| **起動直後（未ロード）** | なし | なし | 最小 | 初回は遅延ロード |
| **デフォルト待機** | `bge-m3` | `qwen3-0.6b` | 低（軽量） | 即座に応答 |
| **Qwen3-VL 使用中** | `qwen3-vl-embedding-2b` | `qwen3-vl-reranker-2b` | 高（重い） | 即座に応答 |
| **Qwen3-VL → フォールバック** | `bge-m3` | `qwen3-0.6b` | 低（解放済） | 即座に応答 |

※ ASR / TTS モデルは上表の遷移対象外で、一度ロードすると常駐します。

---

## 🔧 利用可能なモデル (MLX)

リクエスト時の `model` パラメータで切り替え可能です。未指定時はデフォルトモデルが使用されます。

### Embedding (デフォルト: `bge-m3`)

| モデル ID | Hugging Face モデル | 特徴 |
| :--- | :--- | :--- |
| `gemma-3-300m` | `embeddinggemma-300m-bf16` | 最速。`input_type` に応じたプレフィックスを自動付与 |
| `bge-m3` | `bge-m3-mlx-fp16` | 多言語で堅牢。**日本語 RAG の推奨・デフォルト** |
| `bge-m3-8bit` | `bge-m3-mlx-8bit` | BGE-M3 の 8bit 量子化版。省メモリ |
| `qwen3-0.6b-embed` | `Qwen3-Embedding-0.6B-mxfp8` | Qwen3 Embedding（テキスト専用） |
| `qwen3-vl-embedding-2b` | `Qwen3-VL-Embedding-2B-mxfp8` | 2B マルチモーダル。`instruction` 対応。最高精度・最重量 |

### Reranker (デフォルト: `qwen3-0.6b`)

| モデル ID | Hugging Face モデル | 特徴 |
| :--- | :--- | :--- |
| `qwen3-0.6b` | `Qwen3-Reranker-0.6B-mxfp8` | 生成型クロスエンコーダ（Yes/No ロジット）。高速・高精度 |
| `qwen3-vl-reranker-2b` | `Qwen3-VL-Reranker-2B-mxfp8` | 2B マルチモーダル。`instruction` 対応 |

### Audio (STT デフォルト: `qwen3-asr-1.7b-8bit` / TTS デフォルト: `qwen3-tts-0.6b-base-8bit`)

| モデル ID | Hugging Face モデル | 特徴 |
| :--- | :--- | :--- |
| `qwen3-asr-0.6b-8bit` | `Qwen3-ASR-0.6B-8bit` | 音声認識。最速（リアルタイム比 65.8x） |
| `qwen3-asr-1.7b-8bit` | `Qwen3-ASR-1.7B-8bit` | 音声認識。最高精度（43.9x、デフォルト） |
| `qwen3-tts-0.6b-base-8bit` | `Qwen3-TTS-12Hz-0.6B-Base-8bit` | 音声合成・ボイスクローン。ロードが速い（デフォルト） |
| `qwen3-tts-1.7b-base-8bit` | `Qwen3-TTS-12Hz-1.7B-Base-8bit` | 音声合成。話速が最も安定 |

4bit 版は評価の結果採用を見送り、音声モデルは 8bit に統一しています（[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) §5.6 / §6.5）。

---

## 🎯 モデル選定の指針

ChromaDB に格納した日本語 1000 件での実測（[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) §2〜§4）:

| モデル | 1000件（キャッシュ後） | 1件あたり | 検索精度の所見 |
| :--- | ---: | ---: | :--- |
| `gemma-3-300m` | 2.93 秒 | 約 2.9 ms | 最速。正解文書が Top100 に入らないケースあり |
| `bge-m3` | 6.10 秒 | 約 6.1 ms | 同条件で **Recall 100%**。デフォルト |
| `qwen3-0.6b-embed` | 9.42 秒 | 約 9.4 ms | `gemma-3-300m` と同様に取りこぼしあり |
| `qwen3-vl-embedding-2b` | 37.74 秒 | 約 37.7 ms | Recall 100%。最重量 |

**基本方針**: 検索品質重視なら `bge-m3`（メモリが厳しければ `bge-m3-8bit`）で Embedding し、Top100 を `qwen3-0.6b` でリランク（100 件で約 2 秒）。`gemma-3-300m` は低レイテンシ優先かつ取りこぼしを許容できる用途、Qwen3-VL はオフラインの高精度評価向けです。

---

## 📊 パフォーマンス概要

| タスク | モデル | 実測値 |
| :--- | :--- | :--- |
| Embedding | `bge-m3` | 1000 件を 6.10 秒（キャッシュ後） |
| Rerank | `qwen3-0.6b` | Top100 を約 2.0 秒（約 20 ms/件） |
| STT | `qwen3-asr-1.7b-8bit` | 32.9 秒の音声を 0.75 秒（43.9x リアルタイム） |
| TTS | `qwen3-tts-0.6b-base-8bit` | 平均生成 3.31 秒（4.1x リアルタイム） |

計測条件・全試行の数値・精度比較は [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) を参照してください。

---

## 📦 ディレクトリ構成

```text
embed_reranker/
├── mlx_embed_rerank_server.py   # メインの FastAPI サーバー（API 1235 + ヘルス 1236）
├── run_mlx_server.sh            # 起動・プロセス管理・監視スクリプト
├── pyproject.toml               # 依存関係と uv 設定
├── README.md                    # ドキュメント（英語）
├── README_JA.md                 # ドキュメント（日本語）
├── GEMINI.md                    # AI エージェント向けのプロジェクト context
├── BENCHMARK_REPORT.md          # ベンチマーク結果（Embedding / Rerank / STT / TTS）
├── MIGRATION_SUMMARY.md         # uv / MLX 移行の履歴
├── AUTO_STARTUP_SUMMARY.md      # launchd 自動起動の設定
├── PLAN_STT_TTS_REVIEW.md       # 音声エンドポイントの設計レビュー（実装前の記録）
├── TODO.md                      # 今後の検討事項
├── LICENSE                      # MIT License
├── scripts/
│   ├── rag_pipeline_chromadb.py # ChromaDB 連携の RAG ベンチマーク
│   └── benchmark_100_sentences.py
├── test-tools/                  # 単発確認用スクリプト
│   ├── test_mlx.py
│   └── test_infer.py
└── tests/                       # pytest による結合テスト（設計は tests/TEST_DESIGN.md）
    ├── TEST_DESIGN.md           # テスト設計書（スコープ・ケース一覧・マーカー・保守方針）
    ├── conftest.py              # fixture・マーカー・共通ヘルパー
    ├── test_health.py           # /health（1235）と監視プローブ（1236）
    ├── test_embeddings.py       # POST /v1/embeddings
    ├── test_rerank.py           # POST /v1/rerank・/rerank
    ├── test_audio.py            # POST /v1/audio/*（マーカー: audio）
    ├── test_errors.py           # エラー応答とバリデーション
    └── data/
        ├── test_cases.json      # 全期待値
        └── asr_output.txt
```

---

## 🐍 動作環境

- macOS（Apple Silicon 必須）
- Python **3.13（推奨）**
- Apple MLX
- **ffmpeg**（mp4/m4a/webm 等の音声処理用）: `brew install ffmpeg`
- **Qwen3-VL 系モデルを使う場合のみ `torch` / `torchvision`**
  `pyproject.toml` には含まれていません。未インストールの場合、`transformers` の `AutoImageProcessor` が例外を投げ、`qwen3-vl-embedding-2b` / `qwen3-vl-reranker-2b` へのリクエストは HTTP 500 になります。必要な場合は明示的に導入してください。

  ```bash
  uv pip install torch torchvision
  ```

  それ以外のモデル（bge-m3 / gemma / qwen3 embed・rerank / ASR / TTS）は torch なしで動作します。

---

## 📥 セットアップ

### 1. 依存関係インストール (uv 推奨)

```bash
uv sync
```

---

## ▶️ 起動方法

```bash
./run_mlx_server.sh
```

起動成功時：

```
Health-check server started on port 1236
Uvicorn running on http://0.0.0.0:1235
```

管理サブコマンド:

```bash
./run_mlx_server.sh status    # 稼働状況とロード済みモデル（1235 の /health を参照）
./run_mlx_server.sh restart
./run_mlx_server.sh kill      # ポート 1235 のプロセスを停止
```

### 🛡️ スーパーバイザー監視機能（自動再起動）

起動スクリプト `run_mlx_server.sh` は、フォアグラウンドに常駐するスーパーバイザーとして動作します。

- **ヘルスチェック専用ポート**: MLX 推論は同期的に uvicorn のイベントループをブロックするため、サーバーは別デーモンスレッドで `http.server` ベースの軽量ヘルスサーバーを **ポート 1236** に起動します。スーパーバイザーは 1235 ではなく `http://localhost:1236/health` を監視するため、長時間の TTS / ASR 推論をハングと誤判定しません（経緯: [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) §6.6。TTS の 10 回連続実行で誤再起動が発生した対策）。
- **監視間隔**: 30 秒ごと、curl のタイムアウトは 10 秒。
- **ハング検知と自動再起動**: 「2 回連続」失敗（約 60 秒無応答）でハングと判定し、ポート 1235 を掴んでいるプロセスを停止して再起動します。
- **Grace Period（猶予期間）**: 2 回の猶予により、Qwen3-VL などの重いモデルロード中の誤検知を防ぎます。
- **トレードオフ**: 監視対象は同一プロセス内の別スレッドのため、プロセスが生存していれば `ok` を返します。検知できるのはプロセスの死亡・応答不能であり、イベントループのみが詰まった状態ではありません。

### macOS での自動起動（launchd）

[AUTO_STARTUP_SUMMARY.md](AUTO_STARTUP_SUMMARY.md) を参照してください。`~/Library/LaunchAgents/com.norihito.embed-reranker.plist` がログイン時に本スクリプトを `KeepAlive` 付きで起動します。

---

## 🧪 動作確認（手動）

### Health Check

```bash
curl http://localhost:1235/health   # ロード済みモデルを含む詳細
curl http://localhost:1236/health   # スーパーバイザー用: {"status": "ok"}
```

### Embedding

```bash
curl http://localhost:1235/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "日本語Embeddingのテスト", "input_type": "query"}'
```

#### instruction 指定（Qwen3-VL モデル専用）

```bash
curl http://localhost:1235/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl-embedding-2b",
    "input": ["テキスト1", "テキスト2"],
    "input_type": "document",
    "instruction": "Represent this document for retrieval."
  }'
```

### Rerank

```bash
curl http://localhost:1235/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SATA DOMのリカバリ手順",
    "documents": [
      "SATA DOMを交換してOSを再インストールする手順",
      "メモリ増設の手順",
      "RAID再構築の注意点"
    ],
    "top_k": 2
  }'
```

#### instruction 指定（Qwen3-VL モデル専用）

```bash
curl http://localhost:1235/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl-reranker-2b",
    "query": "猫の写真",
    "documents": ["犬の散歩", "猫の昼寝", "鳥の飛行"],
    "instruction": "Retrieve images or text relevant to the user'"'"'s query."
  }'
```

### 音声認識（STT）

```bash
# 音声ファイルの文字起こし（wav, mp3, mp4, m4a 等）
curl -X POST http://localhost:1235/v1/audio/transcriptions \
  -F file="@sample.wav" \
  -F model="qwen3-asr-1.7b-8bit" \
  -F language="ja"
```

アップロードされたバイト列は拡張子を保持したまま `/tmp` に一時ファイルとして書き出し、そのパスをモデルに渡します（m4a/mp4 は `moov` が末尾配置のため、パイプ入力ではシークできず ffmpeg が空配列を返すことへの対策）。一時ファイルは処理後に必ず削除されます。レスポンスは `{"text": "..."}` です。

### 音声合成（TTS）

```bash
# テキストから音声生成（mp3, wav, flac, ogg）
curl -X POST http://localhost:1235/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "こんにちは。これはテストです。",
    "model": "qwen3-tts-0.6b-base-8bit",
    "voice": "Chelsie",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

`response_format` は mp3 → `audio/mpeg`、wav → `audio/wav`、flac → `audio/flac`、ogg → `audio/ogg` にマップされ、未知の値は `audio/mpeg` にフォールバックします。`voice` / `speed` / `ref_audio` / `ref_text` は指定された場合のみモデルへ渡されます（`speed` は `1.0` 以外のときのみ）。

#### ボイスクローン

```bash
# リファレンス音声から声をクローン
curl -X POST http://localhost:1235/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "これはクローンされた声です。",
    "model": "qwen3-tts-0.6b-base-8bit",
    "ref_audio": "/path/to/reference.wav",
    "ref_text": "リファレンス音声の書き起こし",
    "response_format": "wav"
  }' \
  --output cloned.wav
```

`ref_audio` は**サーバーホスト上のパス**を指定します（クライアントからのアップロードは未実装）。参照音声は **5〜15 秒程度**を推奨します。長すぎると ICL エンコードに時間がかかります（[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) §6.5）。

---

## 🧪 テスト（自動）

サーバーを起動した状態で実行します。

```bash
# 依存関係を含めて同期
uv sync --extra dev

# テスト実行
uv run pytest tests/
```

すべて**稼働中サーバーに対する API 結合テスト**で、モックは使いません。設計方針・ケース一覧・意図的に対象外としている項目は [tests/TEST_DESIGN.md](tests/TEST_DESIGN.md) を参照してください。

| ファイル | 検証内容 |
| :--- | :--- |
| `tests/test_health.py` | `/health` のペイロード構造・デフォルトモデルの実在・ポート 1236 の監視プローブ |
| `tests/test_embeddings.py` | 全 5 モデルの次元数・単位ベクトル性・OpenAI 互換構造・デフォルトモデル・文字列/配列入力の等価性・決定性・`input_type` のプレフィックス挙動・`instruction` が VL 専用であること・意味的健全性 |
| `tests/test_rerank.py` | 両 Reranker のレスポンス構造・降順・期待 1 位・スコア範囲・`/rerank` エイリアス・`top_k`（既定/切り詰め/`0`=全件/超過）・空配列・決定性・`qwen3-0.6b` が `instruction` を無視すること |
| `tests/test_audio.py` | STT の JSON 構造・`/tmp` 一時ファイルの後始末・TTS の Content-Type と RIFF ヘッダ・デフォルトモデル・音声モデルが常駐すること |
| `tests/test_errors.py` | 非対称なエラーコード（embeddings は 500 / rerank・音声は 400）・種別違いモデル・422 バリデーション・404 |

マーカー（`pyproject.toml` に登録済み、`--strict-markers` 有効）:

```bash
uv run pytest tests/                  # 全件
uv run pytest tests/ -m "not audio"   # 遅い STT/TTS を除外
uv run pytest tests/ -m audio         # STT/TTS のみ
```

注意:

- 期待値は `tests/data/test_cases.json` に集約しています。`AVAILABLE_*_MODELS` にモデルを追加・削除した場合は `expected_embed_models` / `expected_rerank_models` / `expected_audio_models` を更新しないと `/health` のアサーションが失敗します。
- `vl` マーカーのテストは `torch` / `torchvision` が未導入なら自動 skip されます（→ [動作環境](#-動作環境)）。
- サーバーが起動していない場合、テストはエラーではなく skip になります。

---

## 💡 RAG 構成例

1. **Embed**: `/v1/embeddings`（`bge-m3`）で文書をベクトル化
2. **Retrieve**: ベクトル DB から上位 50〜100 件を取得
3. **Rerank**: `/v1/rerank`（`qwen3-0.6b`）で上位 10〜20 件に絞り込み
4. **Generate**: 絞り込んだ文書を LLM のコンテキストに渡す

### LLM サーバーとの共存

| 役割 | ポート / Base URL |
| :--- | :--- |
| **LLM サーバー**（LM Studio / Ollama） | `http://localhost:1234/v1` |
| **Embed & Rerank サーバー**（本リポジトリ） | `http://localhost:1235` |
| **ヘルス監視専用**（本リポジトリ） | `http://localhost:1236/health` |

---

## 📜 License / Credits

- **License**: MIT License（詳細は [LICENSE](LICENSE) を参照）
- Models: [mlx-community](https://huggingface.co/mlx-community) / Google / BAAI / Qwen
- Powered by [Apple MLX](https://github.com/ml-explore/mlx) and FastAPI

---

## 📝 変更履歴

### 2026-06-26 — 音声モデルの 8bit 統一とヘルスチェック専用サーバー

- ASR デフォルトを `qwen3-asr-0.6b-8bit` → **`qwen3-asr-1.7b-8bit`** に変更（精度が大幅向上、速度ペナルティは +0.25 秒）
- TTS の 4bit 版（`Qwen3-TTS-12Hz-0.6B-Base-4bit`）を削除し、**8bit に統一**。デフォルトは `qwen3-tts-0.6b-base-8bit`
- MLX 推論中にスーパーバイザーが誤ってプロセスを再起動する問題への対策として、**ポート 1236 のヘルスチェック専用サーバー**をデーモンスレッドで追加。`run_mlx_server.sh` の監視先を 1235 → 1236 に変更
- 詳細な計測結果は [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) を参照

### 2026-06-26 — STT / TTS エンドポイントの追加

- `POST /v1/audio/transcriptions`（OpenAI 互換 STT）と `POST /v1/audio/speech`（OpenAI 互換 TTS、ボイスクローン対応）を追加
- `AVAILABLE_AUDIO_MODELS` と `ModelManager.get_asr()` / `get_tts()` を追加。ロードは `mlx_audio.stt.utils` / `mlx_audio.tts.utils` を使用
- `/health` に `loaded_asr_models` / `loaded_tts_models` / `available_audio` を追加
- 依存関係に `mlx-audio>=0.3.0` と `python-multipart` を追加
- 設計検討の記録は [PLAN_STT_TTS_REVIEW.md](PLAN_STT_TTS_REVIEW.md)（実装時に既定モデル等は変更されています）

### 2026-06-20 — `bge-m3-8bit` の追加

- 省メモリ用途向けに `mlx-community/bge-m3-mlx-8bit` を `AVAILABLE_EMBED_MODELS` へ追加（Embedding は計 5 モデル）

### 2026-06-05 — Qwen3-VL Embedding / Reranker 2B 対応

**追加**: `mlx-community/Qwen3-VL-Embedding-2B-mxfp8` と `mlx-community/Qwen3-VL-Reranker-2B-mxfp8` をサポート。

- `AVAILABLE_EMBED_MODELS` / `AVAILABLE_RERANK_MODELS` に新モデルを登録
- `compute_embeddings` / `compute_rerank` に `instruction` パラメータと `qwen3_vl_*` タイプ対応を追加（`instruction` が効くのは Qwen3-VL 系のみ）
- `EmbReq` / `RerankReq` に `instruction` フィールドを追加
- `mlx_embeddings` が `Qwen3VLProcessor.__init__` をスキップする問題に対し、`AutoProcessor.from_pretrained` から `image_ids` / `video_ids` / `audio_ids` / `chat_template` をコピーする `_fix_qwen3vl_processor` を追加
- **自動フォールバック**: `ModelManager` に 30 秒タイマーを導入。Qwen3-VL モデルは未使用でペアアンロードし、デフォルトモデル（`bge-m3` / `qwen3-0.6b`）へ自動フォールバック
- Qwen3-VL 系の実行には `torch` / `torchvision` が必要（`pyproject.toml` には含まれないため個別導入）

### 2026-04-30 — reranker を mlx_lm に移行

**問題**: `mlx_embeddings.load()` で Qwen3-Reranker を読み込んでいたが、この API は embedding 専用で cross-encoder の `rank()`/`score()` を持たず、`/v1/rerank` が常に 500 エラーを返していた。

**修正**: `mlx_lm.load()` で言語モデルとして正しくロードし、yes/no ロジットスコアリングを実装。実装されているプロンプトは以下の通り（`instruction` は埋め込まれません）。

```python
prompt = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. "
    "Note that the answer can only be \"yes\" or \"no\".\n"
    "<|im_start|>user\n"
    f"<Query>: {query}\n<Document>: {doc}\n"
    "<|im_start|>assistant\n"
)

# 最終トークンの yes/no ロジットから確率を計算
logits = model(input_ids[None, :])
last_logits = logits[0, -1, :]
score = softmax(last_logits[yes_token_id], last_logits[no_token_id])
```

⸻
