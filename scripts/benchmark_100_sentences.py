# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "faker",
#     "httpx",
# ]
# ///

import time
import httpx
from faker import Faker

BASE_URL = "http://localhost:1235"
EMBED_MODEL = "gemma-3-300m"
RERANK_MODEL = "qwen3-0.6b"

def generate_sentences(count: int = 100) -> list[str]:
    fake = Faker("ja_JP")
    sentences = []
    for _ in range(count):
        # 100文字程度の適当な日本語テキストを生成
        text = fake.text(max_nb_chars=100)
        sentences.append(text)
    return sentences

def test_embedding(client: httpx.Client, documents: list[str]) -> list[list[float]]:
    print(f"\n--- 埋め込み (Embedding) テスト ---")
    print(f"モデル: {EMBED_MODEL}")
    print(f"データ数: {len(documents)} 件")
    
    start_time = time.time()
    payload = {
        "model": EMBED_MODEL,
        "input": documents,
        "input_type": "document"
    }
    
    response = client.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=120.0)
    response.raise_for_status()
    
    data = response.json()
    elapsed = time.time() - start_time
    
    print(f"完了時間: {elapsed:.2f} 秒")
    print(f"1件あたりの平均: {(elapsed/len(documents))*1000:.2f} ms")
    
    return [item["embedding"] for item in data["data"]]

def test_rerank(client: httpx.Client, query: str, documents: list[str]):
    print(f"\n--- リランク (Rerank) テスト ---")
    print(f"モデル: {RERANK_MODEL}")
    print(f"クエリ: {query}")
    print(f"対象ドキュメント数: {len(documents)} 件")
    
    start_time = time.time()
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_k": 5
    }
    
    response = client.post(f"{BASE_URL}/v1/rerank", json=payload, timeout=120.0)
    response.raise_for_status()
    
    data = response.json()
    elapsed = time.time() - start_time
    
    print(f"完了時間: {elapsed:.2f} 秒")
    print(f"\n上位5件の結果:")
    for i, result in enumerate(data["results"]):
        idx = result["index"]
        score = result["relevance_score"]
        doc = documents[idx]
        display_doc = doc.replace('\n', ' ')[:60] + "..." if len(doc) > 60 else doc.replace('\n', ' ')
        print(f" {i+1}. [スコア: {score:.4f}] {display_doc}")

def main():
    print("データ生成中...")
    documents = generate_sentences(1000)
    
    # テストとして、クエリに関連する特定の文章を1件仕込みます
    query = "AIや機械学習の最新技術について知りたいです。"
    target_doc = "最新の人工知能(AI)技術と機械学習アルゴリズムの発展により、自然言語処理の精度は飛躍的に向上しています。特に大規模言語モデル(LLM)が注目されています。"
    
    # スライスしても含まれるように前方に仕込みます
    documents[10] = target_doc
    
    print(f"生成完了 (1000件)。サンプルのクエリと関連する文章を1件仕込みました。")
    
    # サーバーのヘルスチェック
    with httpx.Client() as client:
        try:
            health = client.get(f"{BASE_URL}/health", timeout=5.0)
            health.raise_for_status()
            print(f"サーバー接続成功: {health.json()['status']}")
        except Exception as e:
            print(f"エラー: サーバーに接続できません。サーバーが {BASE_URL} で起動しているか確認してください。")
            print(f"詳細: {e}")
            return

        test_embedding(client, documents)
        
        print("\n========== 100件のリランク ==========")
        test_rerank(client, query, documents[:100])
        
        print("\n========== 50件のリランク ==========")
        test_rerank(client, query, documents[:50])

if __name__ == "__main__":
    main()
