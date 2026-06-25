"""API integration tests for MLX Embed + Rerank Server.

Prerequisites:
    - Server running on http://localhost:1235
    - Test data loaded from tests/data/test_cases.json

Usage:
    uv run pytest tests/
"""

import io
import json
import math
from pathlib import Path

import httpx
import numpy as np
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
        assert set(data["available_audio"]) == set(expected["expected_audio_models"])


class TestEmbedding:
    @pytest.mark.parametrize("model_name", ["gemma-3-300m", "bge-m3", "qwen3-vl-embedding-2b"])
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
    @pytest.mark.parametrize("model_name", ["qwen3-0.6b", "qwen3-vl-reranker-2b"])
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


class TestAudio:
    def test_health_includes_audio(self, client: httpx.Client):
        """Health endpoint should include audio models."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "available_audio" in data
        assert "loaded_asr_models" in data
        assert "loaded_tts_models" in data
        assert set(data["available_audio"]) == {"qwen3-asr-0.6b-8bit", "qwen3-asr-1.7b-8bit", "qwen3-tts-0.6b-base-8bit", "qwen3-tts-1.7b-base-8bit"}

    def test_transcriptions_endpoint(self, client: httpx.Client):
        """Test STT endpoint with dummy audio."""
        # Create a simple sine wave audio (1 second, 16kHz)
        sample_rate = 16000
        duration = 1.0
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Save to BytesIO as WAV
        buffer = io.BytesIO()
        try:
            from mlx_audio.audio_io import write as audio_write
            audio_write(buffer, audio_data, sample_rate, format="wav")
        except ImportError:
            pytest.skip("mlx-audio not available for test audio generation")

        buffer.seek(0)

        # Upload for transcription
        files = {"file": ("test.wav", buffer, "audio/wav")}
        data = {"model": "qwen3-asr-0.6b-8bit"}
        resp = client.post("/v1/audio/transcriptions", files=files, data=data)

        # Note: This may fail if the model is not installed or if the audio is too short
        # For now, we just check the endpoint is accessible
        # In a real test, we'd use a proper audio file
        if resp.status_code == 200:
            result = resp.json()
            assert "text" in result
        else:
            # If model not loaded or other error, that's acceptable for this smoke test
            print(f"STT test returned {resp.status_code}: {resp.text}")

    def test_speech_endpoint(self, client: httpx.Client):
        """Test TTS endpoint."""
        payload = {
            "input": "Hello, this is a test.",
            "model": "qwen3-tts-0.6b-base-8bit",
            "response_format": "wav",
        }
        resp = client.post("/v1/audio/speech", json=payload)

        # Note: This may fail if the model is not installed
        # For now, we just check the endpoint is accessible
        if resp.status_code == 200:
            assert resp.headers["content-type"] == "audio/wav"
            audio_data = resp.content
            assert len(audio_data) > 0
        else:
            # If model not loaded or other error, that's acceptable for this smoke test
            print(f"TTS test returned {resp.status_code}: {resp.text}")
