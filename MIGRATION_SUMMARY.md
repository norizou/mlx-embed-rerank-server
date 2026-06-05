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
- **サーバーの起動**: `uv run python ruri_embed_rerank_server.py`
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
