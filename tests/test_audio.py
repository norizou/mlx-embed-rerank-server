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


def _sine_wav_path(tmp_path, sample_rate: int, duration_sec: float, frequency: int) -> str:
    """Same tone, but on disk: `ref_audio` is read by the server, not uploaded.

    The suite drives a server on this same host, so a local path is readable
    by it (documented in README "ref_audio must be a path on the server host").
    """
    path = tmp_path / "ref.wav"
    path.write_bytes(_sine_wav(sample_rate, duration_sec, frequency).getvalue())
    return str(path)


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

    def test_default_model_when_model_omitted(self, client, test_cases, tmp_path):
        """DEFAULT_TTS is Irodori, which has no built-in default voice, so the
        request must still supply a reference (see TestSpeechIrodori)."""
        case = test_cases["audio"]["tts_irodori"]
        ref = _sine_wav_path(
            tmp_path, case["ref_sample_rate"], case["ref_duration_sec"], case["ref_frequency"]
        )
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": "短いテスト",
                "ref_audio": ref,
                "seconds": case["seconds"],
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.headers["content-type"] == "audio/wav"


class TestSpeechIrodori:
    """Irodori is a second TTS engine with its own generate() kwargs.

    It clones from `ref_audio` alone -- no transcript -- and controls length
    with `seconds` / `duration_scale` rather than `speed`. v4.1-Small is a
    unified checkpoint, so cloning, VoiceDesign (`instruct`) and automatic
    duration all come from the one model.
    """

    @pytest.fixture(scope="class")
    def case(self, test_cases):
        return test_cases["audio"]["tts_irodori"]

    def test_voice_clone_returns_wav(self, client, case, tmp_path):
        ref = _sine_wav_path(
            tmp_path, case["ref_sample_rate"], case["ref_duration_sec"], case["ref_frequency"]
        )
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": case["input"],
                "model": case["model"],
                "ref_audio": ref,
                "seconds": case["seconds"],
                "response_format": case["response_format"],
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.headers["content-type"] == case["expected_content_type"]
        assert resp.content[:4] == b"RIFF"

    def test_speed_maps_to_duration_scale(self, client, case, tmp_path):
        """`speed` has no Irodori equivalent; the server inverts it into
        `duration_scale`, so the request must succeed rather than 400."""
        ref = _sine_wav_path(
            tmp_path, case["ref_sample_rate"], case["ref_duration_sec"], case["ref_frequency"]
        )
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": case["input"],
                "model": case["model"],
                "ref_audio": ref,
                "seconds": case["seconds"],
                "speed": 1.2,
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.content[:4] == b"RIFF"

    def test_qwen3_params_are_ignored_not_rejected(self, client, case, tmp_path):
        """`voice` / `ref_text` are logged as ignored, never rejected.

        OpenAI-compatible clients send `voice` unconditionally, so a 400 here
        would break them.
        """
        ref = _sine_wav_path(
            tmp_path, case["ref_sample_rate"], case["ref_duration_sec"], case["ref_frequency"]
        )
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": case["input"],
                "model": case["model"],
                "ref_audio": ref,
                "seconds": case["seconds"],
                "voice": "Chelsie",
                "ref_text": "リファレンス音声の書き起こし",
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.content[:4] == b"RIFF"

    def test_caption_only_voice_design(self, client, case):
        """v4.1 generates from `instruct` alone, with no reference audio."""
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": case["input"],
                "model": case["model"],
                "instruct": case["instruct"],
                "seconds": case["seconds"],
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.content[:4] == b"RIFF"

    def test_ref_audio_and_instruct_combine(self, client, case, tmp_path):
        """Style-controlled cloning: the unified checkpoint takes both at once."""
        ref = _sine_wav_path(
            tmp_path, case["ref_sample_rate"], case["ref_duration_sec"], case["ref_frequency"]
        )
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": case["input"],
                "model": case["model"],
                "ref_audio": ref,
                "instruct": case["instruct"],
                "seconds": case["seconds"],
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.content[:4] == b"RIFF"

    def test_multi_clip_ref_audio(self, client, case, tmp_path):
        """v4 encodes each clip separately and concatenates them."""
        refs = []
        for i, freq in enumerate((case["ref_frequency"], case["ref_frequency"] * 2)):
            path = tmp_path / f"ref_{i}.wav"
            path.write_bytes(
                _sine_wav(case["ref_sample_rate"], case["ref_duration_sec"], freq).getvalue()
            )
            refs.append(str(path))
        resp = client.post(
            "/v1/audio/speech",
            json={
                "input": case["input"],
                "model": case["model"],
                "ref_audio": refs,
                "seconds": case["seconds"],
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.content[:4] == b"RIFF"


class TestHealthReflectsAudioLoads:
    def test_requested_audio_model_appears_in_health(self, client, test_cases):
        case = test_cases["audio"]["stt"]
        buffer = _sine_wav(case["sample_rate"], case["duration_sec"], case["frequency"])
        files = {"file": ("test.wav", buffer, "audio/wav")}
        client.post("/v1/audio/transcriptions", files=files, data={"model": case["model"]})
        loaded = client.get("/health").json()["loaded_asr_models"]
        assert case["model"] in loaded, "ASR models are never auto-unloaded once loaded"
