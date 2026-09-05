"""POST /v1/embeddings."""

import pytest

from conftest import embed, l2_norm, requires_vl

EMBED_MODELS = [
    "gemma-3-300m",
    "bge-m3",
    "bge-m3-8bit",
    "qwen3-0.6b-embed",
    pytest.param("qwen3-vl-embedding-2b", marks=[pytest.mark.vl, requires_vl]),
]


class TestEmbeddingModels:
    @pytest.mark.parametrize("model_name", EMBED_MODELS)
    def test_shape_and_normalization(self, client, test_cases, model_name):
        """Every model returns `expected_dim` floats, L2-normalized to 1.0."""
        case = test_cases["embedding"][model_name]
        data = embed(client, model_name, case["input"], case["input_type"])

        assert data["model"] == model_name
        assert data["object"] == "list"
        assert len(data["data"]) == case["expected_count"]

        for index, item in enumerate(data["data"]):
            assert item["object"] == "embedding"
            assert item["index"] == index
            vector = item["embedding"]
            assert len(vector) == case["expected_dim"]
            assert all(isinstance(v, float) for v in vector)
            assert 0.99 <= l2_norm(vector) <= 1.01, f"not normalized: {l2_norm(vector)}"

    @pytest.mark.parametrize("model_name", EMBED_MODELS)
    def test_openai_compatible_envelope(self, client, test_cases, model_name):
        case = test_cases["embedding"][model_name]
        data = embed(client, model_name, case["input"], case["input_type"])
        assert set(data) == {"object", "data", "model", "usage"}
        assert set(data["usage"]) == {"prompt_tokens", "total_tokens"}


class TestEmbeddingRequestShapes:
    def test_default_model_when_model_omitted(self, client, test_cases):
        """Omitting `model` must resolve to DEFAULT_EMBED."""
        resp = client.post("/v1/embeddings", json={"input": "デフォルトモデルのテスト"})
        assert resp.status_code == 200
        assert resp.json()["model"] == test_cases["embedding_default_model"]

    def test_string_input_equals_single_element_list(self, client, test_cases):
        model = test_cases["embedding_default_model"]
        text = "文字列入力と配列入力の等価性"
        as_string = embed(client, model, text)
        as_list = embed(client, model, [text])
        assert len(as_string["data"]) == 1
        assert as_string["data"][0]["embedding"] == as_list["data"][0]["embedding"]

    def test_deterministic(self, client, test_cases):
        model = test_cases["embedding_default_model"]
        first = embed(client, model, ["同じ入力は同じベクトル"])
        second = embed(client, model, ["同じ入力は同じベクトル"])
        assert first["data"][0]["embedding"] == second["data"][0]["embedding"]


class TestInputTypeHandling:
    """`input_type` only drives the gemma prefix / VL default instruction."""

    def test_gemma_prefix_changes_the_vector(self, client, test_cases):
        assert test_cases["embedding"]["gemma-3-300m"]["prefix_sensitive"] is True
        text = "プレフィックスの効果を確認する文"
        as_query = embed(client, "gemma-3-300m", [text], "query")
        as_document = embed(client, "gemma-3-300m", [text], "document")
        assert as_query["data"][0]["embedding"] != as_document["data"][0]["embedding"]

    @pytest.mark.parametrize("model_name", ["bge-m3", "bge-m3-8bit", "qwen3-0.6b-embed"])
    def test_standard_models_ignore_input_type(self, client, test_cases, model_name):
        assert test_cases["embedding"][model_name]["prefix_sensitive"] is False
        text = "input_type は標準モデルでは無視される"
        as_query = embed(client, model_name, [text], "query")
        as_document = embed(client, model_name, [text], "document")
        assert as_query["data"][0]["embedding"] == as_document["data"][0]["embedding"]

    @pytest.mark.parametrize("model_name", ["bge-m3", "gemma-3-300m"])
    def test_instruction_is_ignored_by_non_vl_models(self, client, model_name):
        """`instruction` is honored only by qwen3-vl-embedding-2b."""
        text = "instruction は VL 以外では無視される"
        without = embed(client, model_name, [text])
        with_instruction = embed(client, model_name, [text], instruction="Totally different instruction.")
        assert without["data"][0]["embedding"] == with_instruction["data"][0]["embedding"]


class TestEmbeddingSemantics:
    """Sanity check on the default model: related text must beat unrelated text."""

    def test_related_text_scores_higher_than_unrelated(self, client, test_cases):
        model = test_cases["embedding_default_model"]
        data = embed(
            client,
            model,
            ["猫がソファで寝ている", "犬が公園を走っている", "量子力学の観測問題について"],
        )
        vectors = [item["embedding"] for item in data["data"]]

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        animal_pair = cosine(vectors[0], vectors[1])
        unrelated_pair = cosine(vectors[0], vectors[2])
        assert animal_pair > unrelated_pair
