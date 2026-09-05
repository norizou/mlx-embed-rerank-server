"""/health (port 1235) and the supervisor probe (port 1236)."""

import httpx
import pytest


class TestHealthEndpoint:
    def test_status_ok(self, health, test_cases):
        assert health["status"] == test_cases["health"]["expected_status"]

    def test_available_models_match_server_config(self, health, test_cases):
        """`available_*` must mirror AVAILABLE_*_MODELS in the server module.

        When a model is added or removed, tests/data/test_cases.json is the
        single place to update.
        """
        expected = test_cases["health"]
        assert set(health["available_embed"]) == set(expected["expected_embed_models"])
        assert set(health["available_rerank"]) == set(expected["expected_rerank_models"])
        assert set(health["available_audio"]) == set(expected["expected_audio_models"])

    def test_loaded_model_lists_present(self, health, test_cases):
        for key in test_cases["health"]["expected_loaded_keys"]:
            assert key in health, f"missing {key}"
            assert isinstance(health[key], list)

    def test_loaded_models_are_a_subset_of_available(self, health):
        available = set(health["available_embed"]) | set(health["available_rerank"]) | set(health["available_audio"])
        loaded = (
            set(health["loaded_embed_models"])
            | set(health["loaded_rerank_models"])
            | set(health["loaded_asr_models"])
            | set(health["loaded_tts_models"])
        )
        assert loaded <= available

    def test_defaults_are_available(self, health, test_cases):
        """The documented defaults must exist in the corresponding model lists."""
        defaults = test_cases["health"]["defaults"]
        assert defaults["embed"] in health["available_embed"]
        assert defaults["rerank"] in health["available_rerank"]
        assert defaults["asr"] in health["available_audio"]
        assert defaults["tts"] in health["available_audio"]

    def test_no_undocumented_fields(self, health):
        """Guards against docs drifting from the payload (e.g. reranker_ready)."""
        assert set(health) == {
            "status",
            "loaded_embed_models",
            "loaded_rerank_models",
            "loaded_asr_models",
            "loaded_tts_models",
            "available_embed",
            "available_rerank",
            "available_audio",
        }


class TestSupervisorProbe:
    """run_mlx_server.sh polls port 1236, not 1235 (see BENCHMARK_REPORT §6.6)."""

    def test_probe_returns_ok(self, probe_client):
        try:
            resp = probe_client.get("/health")
        except httpx.HTTPError as exc:  # pragma: no cover - environment guard
            pytest.skip(f"health probe not reachable: {exc}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_probe_answers_any_path(self, probe_client):
        """The probe is a bare GET handler; the supervisor's URL must not 404."""
        try:
            resp = probe_client.get("/")
        except httpx.HTTPError as exc:  # pragma: no cover - environment guard
            pytest.skip(f"health probe not reachable: {exc}")
        assert resp.status_code == 200
