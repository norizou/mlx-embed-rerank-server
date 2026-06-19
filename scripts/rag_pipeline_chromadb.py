# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "faker",
#     "httpx",
#     "chromadb",
# ]
# ///

import time
import httpx
from faker import Faker
import chromadb
import os

BASE_URL = "http://localhost:1235"
EMBED_MODEL = "qwen3-0.6b-embed"
RERANK_MODEL = "qwen3-0.6b"

def generate_sentences(count: int = 100) -> list[str]:
    fake = Faker("ja_JP")
    sentences = []
    for _ in range(count):
        text = fake.text(max_nb_chars=100)
        sentences.append(text)
    return sentences

def get_embeddings(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "input_type": "document"
    }
    response = client.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=120.0)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]

def get_query_embedding(client: httpx.Client, query: str) -> list[float]:
    payload = {
        "model": EMBED_MODEL,
        "input": [query],
        "input_type": "query"
    }
    response = client.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def rerank_documents(client: httpx.Client, query: str, documents: list[str], top_k: int = 5):
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_k": top_k
    }
    response = client.post(f"{BASE_URL}/v1/rerank", json=payload, timeout=120.0)
    response.raise_for_status()
    return response.json()["results"]

def main():
    print("==================================================")
    print("RAGパイプライン テスト (ChromaDB連携)")
    print("==================================================")
    
    print("\n1. データ生成中...")
    documents = generate_sentences(1000)
    
    query = "AIや機械学習の最新技術について知りたいです。"
    target_doc = "最新の人工知能(AI)技術と機械学習アルゴリズムの発展により、自然言語処理の精度は飛躍的に向上しています。特に大規模言語モデル(LLM)が注目されています。"
    documents[500] = target_doc
    
    print(f"   -> 1000件のダミーデータを生成しました。")
    print(f"   -> 関連する正解データも混入済みです。")
    
    # ChromaDBの初期化
    print("\n2. ChromaDBの初期化...")
    db_path = os.path.abspath("./chroma_db")
    db_client = chromadb.PersistentClient(path=db_path)
    
    collection_name = "test_rag_collection"
    try:
        db_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = db_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   -> コレクション '{collection_name}' を作成しました (Cosine Similarity)。")
    print(f"   -> 保存先ディレクトリ: {db_path}")
    
    with httpx.Client() as api_client:
        print(f"\n3. embed-reranker APIでベクトル化 (モデル: {EMBED_MODEL})...")
        start_time = time.time()
        embeddings = get_embeddings(api_client, documents)
        elapsed = time.time() - start_time
        print(f"   -> 1000件のベクトル化完了: {elapsed:.2f} 秒")
        
        print("\n4. ベクトルデータをChromaDBに保存...")
        start_time = time.time()
        ids = [f"doc_{i}" for i in range(len(documents))]
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        elapsed = time.time() - start_time
        print(f"   -> ChromaDBへの書き込み完了: {elapsed:.2f} 秒")
        
        print(f"\n5. クエリのベクトル化とChromaDB検索 (上位100件を取得)...")
        query_emb = get_query_embedding(api_client, query)
        
        start_time = time.time()
        search_results = collection.query(
            query_embeddings=[query_emb],
            n_results=100
        )
        elapsed = time.time() - start_time
        
        retrieved_docs = search_results["documents"][0]
        print(f"   -> ベクトル検索完了: {elapsed:.4f} 秒")
        print(f"   -> 取得件数: {len(retrieved_docs)} 件")
        
        # 正解データが含まれているかチェック
        target_found = any("最新の人工知能" in doc for doc in retrieved_docs)
        print(f"   -> [検証] Top100件に正解データが含まれているか: {'✅ はい' if target_found else '❌ いいえ'}")
        
        print(f"\n6. 取得した100件をリランク (モデル: {RERANK_MODEL})...")
        start_time = time.time()
        rerank_results = rerank_documents(api_client, query, retrieved_docs, top_k=5)
        elapsed = time.time() - start_time
        print(f"   -> リランク完了: {elapsed:.2f} 秒")
        
        print("\n========== 最終結果 (Top 5) ==========")
        for i, result in enumerate(rerank_results):
            idx = result["index"]
            score = result["relevance_score"]
            doc = retrieved_docs[idx]
            display_doc = doc.replace('\n', ' ')[:60] + "..." if len(doc) > 60 else doc.replace('\n', ' ')
            print(f" {i+1}. [スコア: {score:.4f}] {display_doc}")

if __name__ == "__main__":
    main()
