"""Shared fixtures for the MLX Embed + Rerank Server integration tests.

The suite drives a *running* server (see tests/TEST_DESIGN.md). Nothing is
mocked: every assertion describes behaviour that the deployed server actually
exhibits, so a failure means the server and the documentation disagree.
"""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

DATA_DIR = Path(__file__).parent / "data"
CASES_PATH = DATA_DIR / "test_cases.json"

_CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
BASE_URL = _CASES["server"]["base_url"]
HEALTH_PROBE_URL = _CASES["server"]["health_probe_url"]


def pytest_configure(config):
    config.addinivalue_line("markers", "vl: requires the Qwen3-VL models (needs torch/torchvision)")
    config.addinivalue_line("markers", "audio: exercises the STT/TTS endpoints (slow, loads audio models)")


def torchvision_available() -> bool:
    """Qwen3-VL processors go through transformers' AutoImageProcessor."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None
    )


requires_vl = pytest.mark.skipif(
    not torchvision_available(),
    reason="torch/torchvision not installed; Qwen3-VL models return HTTP 500 (see README Requirements)",
)


@pytest.fixture(scope="session")
def test_cases() -> dict:
    return _CASES


@pytest.fixture(scope="session")
def client(test_cases):
    """HTTP client for the API port (1235).

    Timeout is generous: the first request for a model triggers a lazy load,
    and the 2B Qwen3-VL weights take tens of seconds on a cold cache.
    """
    with httpx.Client(base_url=BASE_URL, timeout=600.0) as c:
        try:
            c.get("/health", timeout=5.0)
        except httpx.HTTPError as exc:  # pragma: no cover - environment guard
            pytest.skip(f"server not reachable at {BASE_URL}: {exc}")
        yield c


@pytest.fixture(scope="session")
def probe_client():
    """HTTP client for the dedicated health port (1236) the supervisor polls."""
    with httpx.Client(base_url=HEALTH_PROBE_URL, timeout=10.0) as c:
        yield c


@pytest.fixture(scope="session")
def health(client) -> dict:
    resp = client.get("/health")
    assert resp.status_code == 200
    return resp.json()


def embed(client, model: str, texts, input_type: str = "document", instruction=None):
    """POST /v1/embeddings and return the parsed body (asserting HTTP 200)."""
    payload = {"model": model, "input": texts, "input_type": input_type}
    if instruction is not None:
        payload["instruction"] = instruction
    resp = client.post("/v1/embeddings", json=payload)
    assert resp.status_code == 200, f"{model}: {resp.status_code} {resp.text[:400]}"
    return resp.json()


def rerank(client, model, query, documents, top_k=None, instruction=None, path="/v1/rerank"):
    """POST /v1/rerank and return the parsed body (asserting HTTP 200)."""
    payload = {"query": query, "documents": documents}
    if model is not None:
        payload["model"] = model
    if top_k is not None:
        payload["top_k"] = top_k
    if instruction is not None:
        payload["instruction"] = instruction
    resp = client.post(path, json=payload)
    assert resp.status_code == 200, f"{model}: {resp.status_code} {resp.text[:400]}"
    return resp.json()


def l2_norm(vector) -> float:
    return sum(v * v for v in vector) ** 0.5
