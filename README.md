# Ruri Embedding & Reranker Server

日本語特化の埋め込みモデル **cl-nagoya/ruri-v3-70m** と  
日本語 CrossEncoder リランカ **cl-nagoya/ruri-v3-reranker-310m** を  
**単一の FastAPI サーバ・単一ポート** で提供する軽量 API サーバです。

LLM 実行基盤（LM Studio 等）とは分離し、  
RAG 用の **Embedding / Rerank 専用エンジン** として動作します。

---

## ✨ 特徴

- ✅ Embedding と Rerank を **1 プロセス / 1 ポート** に統合
- ✅ OpenAI 互換風 API
- ✅ Apple Silicon ネイティブ **MLX** 高速推論
- ✅ 多言語対応（Gemma 3 300M, BGE-M3 等）
- ✅ GGUF 不要
- ✅ OpenWebUI / 自作RAG / LangChain から利用可能

---

## 🧠 提供 API

### ベース URL

http://localhost:1235

### エンドポイント一覧

| Method | Path            | 説明 |
|------|------------------|------|
| GET  | `/health`        | ヘルスチェック |
| POST | `/v1/embeddings` | 日本語埋め込み生成 |
| POST | `/v1/rerank`     | クエリ＋文書の再ランキング |

---

## 🔧 利用可能なモデル (MLX)

リクエスト時の `model` パラメータで切り替え可能です。未指定時はデフォルトモデルが使用されます。

### Embedding (デフォルト: `gemma-3-300m`)
| ID | モデル名 (Hugging Face) | 特徴 |
| :--- | :--- | :--- |
| `gemma-3-300m` | `embedding-gemma-300m-bf16` | 最新 Gemma 3, プレフィックス自動付与 |
| `bge-m3` | `bge-m3-mlx-fp16` | 定番の多言語対応モデル |

### Reranker (デフォルト: `qwen3-0.6b`)
| ID | モデル名 (Hugging Face) | 特徴 |
| :--- | :--- | :--- |
| `qwen3-0.6b` | `Qwen3-Reranker-0.6B-mxfp8` | 生成型 (Yes/No), 高精度 |


---

## 📦 ディレクトリ構成

```
embed_reranker/
├── mlx_embed_rerank_server.py   # メインサーバー
├── run_mlx_server.sh            # 起動・管理スクリプト
├── pyproject.toml               # 依存関係 / uv 設定
├── README.md                    # 本ドキュメント
├── tests/
│   ├── test_api.py              # pytest 統合テスト
│   └── data/
│       └── test_cases.json      # テストケースデータ
└── ...
```

---

## 🐍 動作環境

- Python **3.13（推奨）**
- macOS (Apple Silicon)
- **MLX 搭載**

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
Uvicorn running on http://0.0.0.0:1235
```

---

## 🧪 動作確認（手動）

### Health Check

```bash
curl http://localhost:1235/health
```

### Embedding

```bash
curl http://localhost:1235/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "日本語Embeddingのテスト", "input_type": "query"}'
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

---

## 🧪 テスト（自動）

サーバーを起動した状態で、以下を実行してください。

```bash
# 依存関係を含めて同期
uv sync --extra dev

# テスト実行
uv run pytest tests/
```

テスト内容：
- `/health` — 利用可能モデルの検証
- `/v1/embeddings` — `gemma-3-300m` / `bge-m3` の埋め込み次元・正規化チェック
- `/v1/rerank` — `qwen3-0.6b` / `bge-reranker-v2-m3` のスコア順序・妥当性チェック

テストデータは `tests/data/test_cases.json` で管理しています。


⸻

🧠 想定ユースケース（RAG）
	1.	/v1/embeddings で全文書をベクトル化
	2.	ベクトルDBで上位 50〜100 件取得
	3.	/v1/rerank で top_k=10〜20 に再ランキング
	4.	LLM（LM Studio 等）へ渡す

⸻

🔗 LLM との併用例

役割	URL
LLM (LM Studio)	http://localhost:1234/v1
Embedding / Rerank	http://localhost:1235


⸻

⚠️ 注意事項
	•	Reranker は 文書数に比例して重くなるため、
Embedding で候補を絞ってから使用してください。
	•	GGUF 変換は不要・非対応です。

⸻

## 📜 License / Credits

- Models: [mlx-community](https://huggingface.co/mlx-community) / Google / BAAI
- Powered by [Apple MLX](https://github.com/ml-explore/mlx) / FastAPI

⸻

## 📝 変更履歴

### 2026-04-30 — reranker を mlx_lm に移行

**問題**: `mlx_embeddings.load()` で Qwen3-Reranker を読み込んでいたが、この API は embedding 専用で cross-encoder の `rank()`/`score()` を持たず、`/v1/rerank` が常に 500 エラーを返していた。

**修正**: `mlx_lm.load()` で言語モデルとして正しくロードし、yes/no ロジットスコアリングを実装。

```python
# Qwen3-Reranker 専用プロンプト
prompt = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements...<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# 最終トークンの yes/no ロジットから確率を計算
logits = model(input_ids[None, :])
yes_prob = softmax(logits[0, -1, yes_token_id], logits[0, -1, no_token_id])
```

ヘルスチェックに `reranker_ready` フラグを追加。

⸻


