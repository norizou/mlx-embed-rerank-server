# 次回以降のアップデート向け TODO

## 🔍 日本語対応モデルのMLX版調査と対応検討

仕様検討時（`MLX対応検討.md`）に挙がっていた、日本語の検索精度に優れた有望なモデルについて、最新のMLX版リポジトリの有無や動作検証を行い、次期バージョンでの組み込みを検討する。

### 1. Jina Embeddings シリーズ
- **ターゲット**: `jina-embeddings-v3` または最新の `v4`
- **特徴**: Qwen2.5 / Qwen3 などをバックボーンにした非常に精度の高い多言語・日本語対応 Embedding モデル。
- **タスク**:
  - Hugging Face (`mlx-community` など) に最新版の MLX フォーマットモデルが存在するか調査する。
  - 現在の `mlx_embeddings` ライブラリの `load()` や `generate()` でスムーズに動作するか検証する。

### 2. Japanese Reranker v2 (BGE-M3ベース)
- **ターゲット**: `japanese-bge-reranker-v2-m3` など
- **特徴**: BGE-M3をベースにした日本語特化型の Reranker。
- **タスク**:
  - `mlx-community` に最新の日本語特化 Reranker があるか調査する。
  - 現在の Reranker 実装（`mlx_lm` を用いた Yes/No トークンのロジット確率計算）で正常にスコアリングできるか検証する。
