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
5.  **依存関係の追加**:
    - `torch` / `torchvision` を追加（Vision モデルのロードに必須）。
    - `transformers` を `5.6.2` → `5.10.2` にアップグレード。
