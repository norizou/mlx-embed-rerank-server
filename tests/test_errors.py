"""Error semantics.

These assertions pin down deliberately *asymmetric* behaviour:
`/v1/embeddings` wraps every exception (including the internal 400 for an
unknown model) into a 500 with a traceback, while the other endpoints let the
400 through. Documented in README.md "Error responses".
"""

import io

import pytest


@pytest.fixture(scope="module")
def unknown_model(test_cases):
    return test_cases["errors"]["unknown_model_name"]


@pytest.fixture(scope="module")
def expected_status(test_cases):
    return test_cases["errors"]["expected_status"]


class TestUnknownModel:
    def test_embeddings_wraps_the_400_into_a_500(self, client, unknown_model, expected_status):
        resp = client.post("/v1/embeddings", json={"model": unknown_model, "input": "x"})
        assert resp.status_code == expected_status["embeddings_unknown_model"]
        assert unknown_model in resp.json()["detail"]

    def test_rerank_returns_400(self, client, unknown_model, expected_status):
        resp = client.post(
            "/v1/rerank", json={"model": unknown_model, "query": "q", "documents": ["a"]}
        )
        assert resp.status_code == expected_status["rerank_unknown_model"]
        assert resp.json()["detail"] == f"Unsupported rerank model: {unknown_model}"

    def test_stt_returns_400(self, client, unknown_model, expected_status):
        files = {"file": ("dummy.wav", io.BytesIO(b"RIFF0000WAVEfmt "), "audio/wav")}
        resp = client.post(
            "/v1/audio/transcriptions", files=files, data={"model": unknown_model}
        )
        assert resp.status_code == expected_status["stt_unknown_model"]
        assert resp.json()["detail"] == f"Unsupported ASR model: {unknown_model}"

    def test_tts_returns_400(self, client, unknown_model, expected_status):
        resp = client.post("/v1/audio/speech", json={"input": "x", "model": unknown_model})
        assert resp.status_code == expected_status["tts_unknown_model"]
        assert resp.json()["detail"] == f"Unsupported TTS model: {unknown_model}"


class TestWrongModelType:
    """An audio model of the wrong kind is rejected before any load happens."""

    def test_tts_model_rejected_by_stt_endpoint(self, client, test_cases, expected_status):
        model = test_cases["errors"]["wrong_type"]["tts_model_for_stt"]
        files = {"file": ("dummy.wav", io.BytesIO(b"RIFF0000WAVEfmt "), "audio/wav")}
        resp = client.post("/v1/audio/transcriptions", files=files, data={"model": model})
        assert resp.status_code == expected_status["stt_wrong_type_model"]
        assert resp.json()["detail"] == f"Model {model} is not an ASR model"

    def test_asr_model_rejected_by_tts_endpoint(self, client, test_cases, expected_status):
        model = test_cases["errors"]["wrong_type"]["asr_model_for_tts"]
        resp = client.post("/v1/audio/speech", json={"input": "x", "model": model})
        assert resp.status_code == expected_status["tts_wrong_type_model"]
        assert resp.json()["detail"] == f"Model {model} is not a TTS model"


class TestTTSEngineParameters:
    """Qwen3-TTS and Irodori take different generate() kwargs.

    Every assertion here is rejected *before* `get_tts()` runs, so none of
    these tests pull weights -- that is why they live outside the `audio` mark.
    """

    @pytest.fixture(scope="class")
    def engines(self, test_cases):
        return test_cases["errors"]["wrong_engine"]

    def test_irodori_only_param_rejected_by_qwen3(self, client, engines, expected_status):
        model = engines["qwen3_model"]
        resp = client.post(
            "/v1/audio/speech", json={"input": "x", "model": model, "seconds": 4.0}
        )
        assert resp.status_code == expected_status["tts_wrong_engine_param"]
        detail = resp.json()["detail"]
        assert "seconds" in detail and model in detail

    def test_max_seconds_rejected_by_qwen3(self, client, engines, expected_status):
        """max_seconds は Irodori の duration クランプ用で Qwen3 には無い。"""
        model = engines["qwen3_model"]
        resp = client.post(
            "/v1/audio/speech", json={"input": "x", "model": model, "max_seconds": 60}
        )
        assert resp.status_code == expected_status["tts_wrong_engine_param"]
        assert "max_seconds" in resp.json()["detail"]

    def test_multi_clip_ref_audio_rejected_by_qwen3(self, client, engines, expected_status):
        """A list of reference clips is an Irodori v4 feature; Qwen3 takes one path."""
        model = engines["qwen3_model"]
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "x", "model": model, "ref_audio": ["/a.wav", "/b.wav"]},
        )
        assert resp.status_code == expected_status["tts_wrong_engine_param"]
        assert "list" in resp.json()["detail"]

    def test_irodori_default_without_reference_rejected(self, client, expected_status):
        """DEFAULT_TTS is Irodori, which has no built-in default voice (unlike
        Qwen3's `voice` presets); omitting both ref_audio and instruct must 400
        rather than silently return an unconditioned, undefined voice."""
        resp = client.post("/v1/audio/speech", json={"input": "x"})
        assert resp.status_code == expected_status["tts_irodori_missing_reference"]
        detail = resp.json()["detail"]
        assert "ref_audio" in detail and "instruct" in detail


class TestRequestValidation:
    """FastAPI/Pydantic validation happens before any model is touched."""

    def test_embeddings_requires_input(self, client):
        assert client.post("/v1/embeddings", json={}).status_code == 422

    def test_rerank_requires_query_and_documents(self, client):
        assert client.post("/v1/rerank", json={"query": "q"}).status_code == 422
        assert client.post("/v1/rerank", json={"documents": ["a"]}).status_code == 422

    def test_speech_requires_input(self, client):
        assert client.post("/v1/audio/speech", json={}).status_code == 422

    def test_transcriptions_requires_file(self, client):
        assert client.post("/v1/audio/transcriptions", data={}).status_code == 422

    def test_unknown_route_is_404(self, client):
        assert client.get("/v1/nope").status_code == 404
