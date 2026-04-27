# Ruri Embedding & Reranker Server

日本語特化の埋め込みモデル **cl-nagoya/ruri-v3-70m** と  
日本語 CrossEncoder リランカ **cl-nagoya/ruri-v3-reranker-310m** を  
**単一の FastAPI サーバ・単一ポート** で提供する軽量 API サーバです。

LLM 実行基盤（LM Studio 等）とは分離し、  
RAG 用の **Embedding / Rerank 専用エンジン** として動作します。

---

## ✨ 特徴

- ✅ 日本語特化モデル（ruri-v3 系）を使用
- ✅ Embedding と Rerank を **1 プロセス / 1 ポート** に統合
- ✅ OpenAI 互換風 API
- ✅ Hugging Face (transformers / sentence-transformers) 直利用
- ✅ GGUF 不要
- ✅ macOS Apple Silicon で **MPS 対応**
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

## 🔧 使用モデル

### Embedding
- **Model**: `cl-nagoya/ruri-v3-70m`
- **方式**: Mean Pooling + L2 Normalization
- **用途**: ベクトル検索 / RAG 事前検索

### Reranker
- **Model**: `cl-nagoya/ruri-v3-reranker-310m`
- **方式**: CrossEncoder
- **用途**: Embedding 上位候補の再順位付け

---

## 📦 ディレクトリ構成（例）

embed_reranker/
├─ ruri_embed_rerank_server.py
├─ requirements.txt
├─ README.md
└─ .venv/

---

## 🐍 動作環境

- Python **3.11（推奨）**
- macOS (Apple Silicon) / Linux
- CPU または MPS（自動判定）

---

## 📥 セットアップ

### 1. 仮想環境作成

```bash
python3.11 -m venv .venv
source .venv/bin/activate

2. 依存関係インストール

pip install -U pip
pip install -r requirements.txt

requirements.txt（例）

fastapi>=0.110
uvicorn[standard]>=0.27
transformers>=4.41
sentence-transformers>=3.0
torch
numpy
sentencepiece
protobuf


⸻

▶️ 起動方法

uvicorn ruri_embed_rerank_server:app --host 0.0.0.0 --port 1235

起動成功時：

Uvicorn running on http://0.0.0.0:1235


⸻

🧪 動作確認

Health Check

curl http://localhost:1235/health

{
  "status": "ok",
  "device": "mps",
  "embedding_model": "cl-nagoya/ruri-v3-70m",
  "rerank_model": "cl-nagoya/ruri-v3-reranker-310m"
}


⸻

Embedding

curl http://localhost:1235/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"日本語Embeddingのテスト"}'


⸻

Rerank

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

📜 License / Credits
	•	Models by cl-nagoya
	•	Powered by Hugging Face / PyTorch / FastAPI

⸻


