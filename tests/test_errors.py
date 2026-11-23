"""Error semantics.

Request errors are 400 on every endpoint. `/v1/embeddings` validates the model
name, image inputs and `dimensions` before loading anything, and only wraps
*unexpected* exceptions into a 500 with a traceback. (It used to turn the
unknown-model 400 into a 500 as well.) Documented in README.md "Error responses".
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
    def test_embeddings_returns_400(self, client, unknown_model, expected_status):
        resp = client.post("/v1/embeddings", json={"model": unknown_model, "input": "x"})
        assert resp.status_code == expected_status["embeddings_unknown_model"]
        assert resp.json()["detail"] == f"Unsupported embedding model: {unknown_model}"

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


class TestEmbeddingInputErrors:
    """Image / `dimensions` problems are 400s raised before the model is loaded."""

    TINY_PNG = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )

    def test_image_on_text_only_model(self, client, test_cases):
        model = test_cases["embedding_images"]["text_only_model"]
        resp = client.post("/v1/embeddings", json={"model": model, "input": [{"type": "image", "image": self.TINY_PNG}]})
        assert resp.status_code == test_cases["errors"]["expected_status"]["embeddings_image_on_text_model"]
        assert "does not accept image inputs" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "image",
        ["https://example.com/cat.png", "/etc/hosts", "data:image/png;base64,not-base64!!", "data:image/png,rawbytes"],
    )
    def test_rejects_urls_paths_and_bad_base64(self, client, test_cases, image):
        """Only base64 data is accepted: URLs and paths would let callers make the server fetch or read files."""
        model = test_cases["embedding_images"]["model"]
        resp = client.post("/v1/embeddings", json={"model": model, "input": [{"type": "image", "image": image}]})
        assert resp.status_code == test_cases["errors"]["expected_status"]["embeddings_bad_image"]

    def test_unknown_object_shape(self, client, test_cases):
        model = test_cases["embedding_images"]["model"]
        resp = client.post("/v1/embeddings", json={"model": model, "input": [{"type": "audio", "audio": "x"}]})
        assert resp.status_code == test_cases["errors"]["expected_status"]["embeddings_bad_image"]

    def test_dimensions_on_non_matryoshka_model(self, client, test_cases):
        case = test_cases["embedding_matryoshka"]
        if not case["non_matryoshka_model"]:
            pytest.skip("every registered embedding model supports `dimensions`")
        resp = client.post("/v1/embeddings", json={"model": case["non_matryoshka_model"], "input": "x", "dimensions": 256})
        assert resp.status_code == test_cases["errors"]["expected_status"]["embeddings_bad_dimensions"]

    @pytest.mark.parametrize("dimensions", [0, -1, 4096])
    def test_dimensions_out_of_range(self, client, test_cases, dimensions):
        case = test_cases["embedding_matryoshka"]
        resp = client.post("/v1/embeddings", json={"model": case["model"], "input": "x", "dimensions": dimensions})
        assert resp.status_code == test_cases["errors"]["expected_status"]["embeddings_bad_dimensions"]
