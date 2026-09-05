"""POST /v1/audio/transcriptions and POST /v1/audio/speech.

Marked `audio`: these load Qwen3-ASR / Qwen3-TTS weights and are the slowest
part of the suite. Skip them with `-m "not audio"`.
"""

import io

import numpy as np
import pytest

pytestmark = pytest.mark.audio


def _sine_wav(sample_rate: int, duration_sec: float, frequency: int) -> io.BytesIO:
    """Synthesize a WAV buffer without touching the filesystem."""
    audio_write = pytest.importorskip("mlx_audio.audio_io").write
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    samples = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    buffer = io.BytesIO()
    audio_write(buffer, samples, sample_rate, format="wav")
    buffer.seek(0)
    return buffer


class TestTranscriptions:
    def test_returns_text_field(self, client, test_cases):
        """A short tone carries no speech; the contract is the JSON shape."""
        case = test_cases["audio"]["stt"]
        buffer = _sine_wav(case["sample_rate"], case["duration_sec"], case["frequency"])
        files = {"file": ("test.wav", buffer, "audio/wav")}
        resp = client.post(
            "/v1/audio/transcriptions",
            files=files,
            data={"model": case["model"], "language": case["language"]},
        )
        assert resp.status_code == 200, resp.text[:400]
        body = resp.json()
        assert set(body) == {"text"}
        assert isinstance(body["text"], str)

    def test_default_model_when_model_omitted(self, client, test_cases):
        case = test_cases["audio"]["stt"]
        buffer = _sine_wav(case["sample_rate"], case["duration_sec"], case["frequency"])
        files = {"file": ("test.wav", buffer, "audio/wav")}
        resp = client.post("/v1/audio/transcriptions", files=files)
        assert resp.status_code == 200, resp.text[:400]
        assert "text" in resp.json()

    def test_temporary_file_is_cleaned_up(self, client, test_cases, tmp_path):
        """The endpoint writes /tmp/stt_<uuid><ext> and must remove it."""
        import glob

        before = set(glob.glob("/tmp/stt_*"))
        case = test_cases["audio"]["stt"]
        buffer = _sine_wav(case["sample_rate"], case["duration_sec"], case["frequency"])
        files = {"file": ("test.wav", buffer, "audio/wav")}
        client.post("/v1/audio/transcriptions", files=files, data={"model": case["model"]})
        assert set(glob.glob("/tmp/stt_*")) <= before


class TestSpeech:
    def test_returns_audio_bytes(self, client, test_cases):
        case = test_cases["audio"]["tts"]
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": case["input"],
                "model": case["model"],
                "response_format": case["response_format"],
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.headers["content-type"] == case["expected_content_type"]
        assert resp.headers["content-disposition"].endswith(
            f"filename=speech.{case['response_format']}"
        )
        assert len(resp.content) > 0

    def test_wav_output_has_riff_header(self, client, test_cases):
        case = test_cases["audio"]["tts"]
        resp = client.post(
            "/v1/audio/speech",
            json={"input": case["input"], "model": case["model"], "response_format": "wav"},
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.content[:4] == b"RIFF"

    def test_default_model_when_model_omitted(self, client, test_cases):
        """DEFAULT_TTS is the model the benchmark selected."""
        resp = client.post("/v1/audio/speech", json={"input": "短いテスト", "response_format": "wav"})
        assert resp.status_code == 200, resp.text[:400]
        assert resp.headers["content-type"] == "audio/wav"


class TestHealthReflectsAudioLoads:
    def test_requested_audio_model_appears_in_health(self, client, test_cases):
        case = test_cases["audio"]["stt"]
        buffer = _sine_wav(case["sample_rate"], case["duration_sec"], case["frequency"])
        files = {"file": ("test.wav", buffer, "audio/wav")}
        client.post("/v1/audio/transcriptions", files=files, data={"model": case["model"]})
        loaded = client.get("/health").json()["loaded_asr_models"]
        assert case["model"] in loaded, "ASR models are never auto-unloaded once loaded"
