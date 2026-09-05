# uvへの移行作業まとめ

既存の `pip` ベースの環境から `uv` を使用したプロジェクト管理環境へ移行しました。

## 実施内容

1. **既存環境のクリーンアップ**
   - 以前の仮想環境（`.venv` ディレクトリ）を削除しました。

2. **uvプロジェクトの初期化**
   - `uv init .` を実行し、プロジェクト構成（`pyproject.toml`）を生成しました。
   - `uv init` によって自動生成された不要な `main.py` を削除しました。

3. **依存関係の移行**
   - `requirements.txt` に記載されていた以下のライブラリを `uv add` を使用して追加しました。
     - `fastapi>=0.110`
     - `uvicorn[standard]>=0.27`
     - `transformers>=4.41`
     - `sentence-transformers>=3.0`
     - `torch`
     - `numpy`
     - `sentencepiece`
     - `protobuf`
   - これにより、`pyproject.toml` と `uv.lock` が生成され、依存関係が厳密に管理されるようになりました。

4. **旧ファイルの削除**
   - 移行が完了したため、不要となった `requirements.txt` を削除しました。

## 今後の使用方法

- **パッケージの追加**: `uv add <package_name>`
- **パッケージの削除**: `uv remove <package_name>`
- **サーバーの起動**: `./run_mlx_server.sh` または `uv run uvicorn mlx_embed_rerank_server:app --host 0.0.0.0 --port 1235`
- **環境の同期**: `uv sync`

---

## 追記：MLXバックエンドへの移行 (2026-04-30)

Apple Silicon環境でのパフォーマンスを最大化するため、PyTorch(MPS)ベースのRuriから、MLXベースのサーバへ移行しました。

### 変更点
1.  **推論エンジンの変更**: MLXフレームワークを採用。
2.  **モデルの刷新**:
    - Embedding: `Gemma 3 300M (bf16)` - プレフィックスによるタスク最適化に対応。
    - Reranker: `Qwen3-Reranker-0.6B (mxfp8)` - 生成型Yes/Noスコアリングによる高精度化。
3.  **マルチモデル対応**: リクエストパラメータでモデルを選択できる管理機能を実装。
4.  **管理スクリプトの強化**: `run_mlx_server.sh` に `status`, `kill`, `restart` 機能を追加。

---

## 追記：Qwen3-VL モデル対応 (2026-06-05)

Vision Language Model (VLM) ベースの Embedding / Reranker を追加し、vlm-eval プロジェクトのモデルをサーバー経由で利用できるようにしました。

### 変更点
1.  **モデルの追加**:
    - Embedding: `Qwen3-VL-Embedding-2B-mxfp8` - マルチモーダル, instruction 対応。
    - Reranker: `Qwen3-VL-Reranker-2B-mxfp8` - マルチモーダル, instruction 対応。
2.  **API の拡張**:
    - `EmbReq` / `RerankReq` に `instruction` パラメータを追加。
    - `input_type` による query/document の区別と instruction の自動付与に対応。
3.  **ワークアラウンドの追加**:
    - `mlx_embeddings` の `Processor.from_pretrained` が `Qwen3VLProcessor.__init__` をスキップする問題に対し、`_fix_qwen3vl_processor` で `image_ids` / `video_ids` / `audio_ids` / `chat_template` を `AutoProcessor.from_pretrained` からコピーして修復。
4.  **自動フォールバック（メモリ最適化）**:
    - `ModelManager` に 30 秒間隔の `_check_inactivity` タイマーを追加。
    - Qwen3-VL 系モデルは 30 秒未使用で **ペアアンロード**（embed / rerank 両方同時解放）。
    - アンロードと同時にデフォルトモデル（`bge-m3` / `qwen3-0.6b`）をプリロード。
    - アンロード時に `mx.metal.clear_cache()` を実行し GPU メモリを即座に解放。
5.  **依存関係の扱い**:
    - Qwen3-VL 系のロードには `torch` / `torchvision` が必要（`transformers` の `AutoImageProcessor` 経由）。
      ただしこれらは `pyproject.toml` の依存には**含めていない**（他モデルは torch 不要で動くため）。
      Qwen3-VL を使う環境でのみ `uv pip install torch torchvision` を個別実行する。未導入の場合、
      `qwen3-vl-*` へのリクエストは `AutoImageProcessor requires the Torchvision library` により HTTP 500 になる。
    - `transformers` は `mlx-embeddings` の推移的依存として導入される（`pyproject.toml` には明示していない）。

---

## 追記：STT / TTS（音声）対応と 8bit 統一 (2026-06-26)

Qwen3-ASR / Qwen3-TTS を同一プロセスに統合し、OpenAI 互換の音声エンドポイントを追加しました。

### 変更点
1.  **エンドポイントの追加**:
    - `POST /v1/audio/transcriptions`（STT、multipart/form-data）
    - `POST /v1/audio/speech`（TTS、ボイスクローン対応、mp3/wav/flac/ogg 出力）
2.  **モデル管理の拡張**: `AVAILABLE_AUDIO_MODELS` と `ModelManager.get_asr()` / `get_tts()` を追加。ロードは `mlx_audio.stt.utils` / `mlx_audio.tts.utils` を使用。音声モデルは自動アンロードの対象外。
3.  **依存関係の追加**: `mlx-lm`、`mlx-audio>=0.3.0`、`python-multipart` を `pyproject.toml` に追加。mp4/m4a/webm の入力にはシステムの `ffmpeg` が必要。
4.  **量子化の統一**: ベンチマークの結果、4bit 版は精度・安定性で劣るため削除し、ASR / TTS とも 8bit に統一。デフォルトは ASR `qwen3-asr-1.7b-8bit`、TTS `qwen3-tts-0.6b-base-8bit`。
5.  **ヘルスチェック専用サーバー**: MLX 推論が uvicorn のイベントループをブロックし、スーパーバイザーに誤って再起動される問題への対策として、ポート `1236` に `http.server` ベースの軽量ヘルスサーバーをデーモンスレッドで追加。`run_mlx_server.sh` の監視先を 1235 → 1236 に変更。
6.  **Embedding モデルの追加**: 省メモリ用途向けに `bge-m3-8bit` を追加（Embedding は計 5 モデル）。

計測結果と選定根拠は `BENCHMARK_REPORT.md` を参照。

---

## 補足：パスについて

本ドキュメント中のパス例 `/Users/norihito/AI/embed_reranker` は移行当時のものです。
現在のリポジトリ配置は `/Users/norihito/Projects/AI/Workspace/embed_reranker` で、
launchd の plist もこのパスを参照しています。
