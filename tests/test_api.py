"""API integration tests for MLX Embed + Rerank Server.

Prerequisites:
    - Server running on http://localhost:1235
    - Test data loaded from tests/data/test_cases.json

Usage:
    uv run pytest tests/
"""

import json
import math
from pathlib import Path

import httpx
import pytest

BASE_URL = "http://localhost:1235"
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
        yield c


@pytest.fixture(scope="session")
def test_cases():
    path = DATA_DIR / "test_cases.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestHealth:
    def test_health_endpoint(self, client: httpx.Client, test_cases: dict):
        expected = test_cases["health"]
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == expected["expected_status"]
        assert set(data["available_embed"]) == set(expected["expected_embed_models"])
        assert set(data["available_rerank"]) == set(expected["expected_rerank_models"])


class TestEmbedding:
    @pytest.mark.parametrize("model_name", ["gemma-3-300m", "bge-m3"])
    def test_embedding_models(self, client: httpx.Client, test_cases: dict, model_name: str):
        case = test_cases["embedding"][model_name]
        payload = {
            "model": model_name,
            "input": case["input"],
            "input_type": case["input_type"],
        }
        resp = client.post("/v1/embeddings", json=payload)
        assert resp.status_code == 200, f"{model_name} embedding failed: {resp.text}"
        data = resp.json()

        assert data["model"] == model_name
        assert data["object"] == "list"
        assert len(data["data"]) == case["expected_count"]

        for item in data["data"]:
            assert item["object"] == "embedding"
            emb = item["embedding"]
            assert len(emb) == case["expected_dim"]
            # Validate normalized vector (unit length ~1.0)
            norm = math.sqrt(sum(v * v for v in emb))
            assert 0.99 <= norm <= 1.01, f"Embedding not normalized: {norm}"


class TestRerank:
    @pytest.mark.parametrize("model_name", ["qwen3-0.6b"])
    def test_rerank_models(self, client: httpx.Client, test_cases: dict, model_name: str):
        case = test_cases["rerank"][model_name]
        payload = {
            "model": model_name,
            "query": case["query"],
            "documents": case["documents"],
            "top_k": case["top_k"],
        }
        resp = client.post("/v1/rerank", json=payload)
        assert resp.status_code == 200, f"{model_name} rerank failed: {resp.text}"
        data = resp.json()

        assert data["model"] == model_name
        results = data["results"]
        assert len(results) == min(case["top_k"], len(case["documents"]))

        # Scores should be in descending order
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

        # Sanity: all scores are finite numbers
        for s in scores:
            assert isinstance(s, float)
            assert not math.isnan(s)

        # Check that the most relevant document is as expected
        if "expected_first_index" in case:
            assert results[0]["index"] == case["expected_first_index"]
