"""POST /v1/embeddings."""

import pytest

from conftest import embed, l2_norm, requires_vl

EMBED_MODELS = [
    "gemma-3-300m",
    "embeddinggemma-2",
    "qwen3-0.6b-embed",
    # bge-m3 / bge-m3-8bit are disabled in AVAILABLE_EMBED_MODELS (cache removed 2026-11-24).
    # qwen3-vl-embedding-2b is disabled in AVAILABLE_EMBED_MODELS (not cached, needs torch):
    # pytest.param("qwen3-vl-embedding-2b", marks=[pytest.mark.vl, requires_vl]),
]


STANDARD_MODELS: list[str] = []  # e.g. ["bge-m3", "bge-m3-8bit"] when re-enabled


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
    """`input_type` drives the gemma / embeddinggemma-2 prompt, the qwen3-0.6b-embed
    query instruction and the VL default instruction; BGE-M3 ignores it."""

    @pytest.mark.parametrize("model_name", ["gemma-3-300m", "embeddinggemma-2", "qwen3-0.6b-embed"])
    def test_query_prompt_changes_the_vector(self, client, test_cases, model_name):
        assert test_cases["embedding"][model_name]["prefix_sensitive"] is True
        text = "プレフィックスの効果を確認する文"
        as_query = embed(client, model_name, [text], "query")
        as_document = embed(client, model_name, [text], "document")
        assert as_query["data"][0]["embedding"] != as_document["data"][0]["embedding"]

    def test_qwen3_embed_honors_instruction_on_queries(self, client):
        text = "instruction は qwen3-0.6b-embed のクエリに効く"
        default = embed(client, "qwen3-0.6b-embed", [text], "query")
        custom = embed(client, "qwen3-0.6b-embed", [text], "query", instruction="Retrieve semantically similar text.")
        assert default["data"][0]["embedding"] != custom["data"][0]["embedding"]

    # `standard` (CLS) models; empty while bge-m3 / bge-m3-8bit are disabled, so this test is skipped.
    @pytest.mark.parametrize("model_name", STANDARD_MODELS)
    def test_standard_models_ignore_input_type(self, client, test_cases, model_name):
        assert test_cases["embedding"][model_name]["prefix_sensitive"] is False
        text = "input_type は標準モデルでは無視される"
        as_query = embed(client, model_name, [text], "query")
        as_document = embed(client, model_name, [text], "document")
        assert as_query["data"][0]["embedding"] == as_document["data"][0]["embedding"]

    @pytest.mark.parametrize("model_name", ["gemma-3-300m", "embeddinggemma-2"])
    def test_instruction_is_ignored_by_prompt_models(self, client, model_name):
        """`instruction` is honored only by qwen3-0.6b-embed (queries) and qwen3-vl-embedding-2b."""
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


def _cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestRetrievalSanity:
    """Pooling regression guard.

    The server used to take the first token (CLS) for every model. That is right
    for BGE-M3 but wrong for EmbeddingGemma (mean pooling + dense) and
    Qwen3-Embedding (last-token pooling): with it, those models ranked the answer
    below unrelated passages (JMTEB nDCG@10 ~3-18 instead of ~75-99). The
    animal/quantum check above did not catch it, and neither does a single query:
    each of four queries must rank its own answer first (the CLS path fails this).
    """

    @pytest.mark.parametrize(
        "model_name", ["gemma-3-300m", "embeddinggemma-2", "qwen3-0.6b-embed"]
    )
    def test_each_query_ranks_its_answer_first(self, client, test_cases, model_name):
        case = test_cases["embedding_retrieval"]
        queries = [d["embedding"] for d in embed(client, model_name, case["queries"], "query")["data"]]
        docs = [d["embedding"] for d in embed(client, model_name, case["documents"], "document")["data"]]
        top = [max(range(len(docs)), key=lambda j: _cosine(q, docs[j])) for q in queries]
        assert top == list(range(len(queries))), f"{model_name}: top documents {top}"


class TestMatryoshka:
    def test_dimensions_truncates_and_renormalizes(self, client, test_cases):
        case = test_cases["embedding_matryoshka"]
        text = "Matryoshka で次元を切り詰める"
        full = embed(client, case["model"], [text])["data"][0]["embedding"]
        resp = client.post(
            "/v1/embeddings", json={"model": case["model"], "input": [text], "dimensions": case["dimensions"]}
        )
        assert resp.status_code == 200, resp.text[:300]
        short = resp.json()["data"][0]["embedding"]
        assert len(short) == case["dimensions"]
        assert 0.99 <= l2_norm(short) <= 1.01
        head = full[: case["dimensions"]]
        norm = l2_norm(head)
        assert all(abs(a - b / norm) < 1e-4 for a, b in zip(short, head))


def png_data_uri(rgb, size=(64, 64)) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def text_page_data_uri(size=(600, 840)) -> str:
    """A page-like image (rows of small black boxes on white), so the resolution matters."""
    import base64
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for row, y in enumerate(range(20, size[1] - 20, 18)):
        for x in range(20 + (row % 3) * 4, size[0] - 20, 12):
            if (x // 12 + row) % 5:
                draw.rectangle((x, y, x + 8, y + 10), fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class TestImageInputs:
    """embeddinggemma-2 embeds images into the same space as text."""

    def test_image_url_object_returns_normalized_vector(self, client, test_cases):
        case = test_cases["embedding_images"]
        item = {"type": "image_url", "image_url": {"url": png_data_uri((200, 30, 30))}}
        resp = client.post("/v1/embeddings", json={"model": case["model"], "input": [item]})
        assert resp.status_code == 200, resp.text[:300]
        vector = resp.json()["data"][0]["embedding"]
        assert len(vector) == case["expected_dim"]
        assert 0.99 <= l2_norm(vector) <= 1.01

    def test_mixed_text_and_image_keep_order(self, client, test_cases):
        """A mixed list returns one vector per item, in input order, equal to separate calls."""
        model = test_cases["embedding_images"]["model"]
        img = {"type": "image", "image": png_data_uri((30, 30, 200))}
        mixed = client.post("/v1/embeddings", json={"model": model, "input": ["青い画像", img]})
        assert mixed.status_code == 200, mixed.text[:300]
        mixed = mixed.json()["data"]
        alone_text = embed(client, model, ["青い画像"])["data"][0]["embedding"]
        alone_img = client.post("/v1/embeddings", json={"model": model, "input": [img]}).json()["data"][0]["embedding"]
        assert [d["index"] for d in mixed] == [0, 1]
        assert _cosine(mixed[0]["embedding"], alone_text) > 0.999
        assert _cosine(mixed[1]["embedding"], alone_img) > 0.999

    def test_caption_matches_its_color(self, client, test_cases):
        """Text and image share one space: each caption must be closest to its own color."""
        model = test_cases["embedding_images"]["model"]
        images = [{"type": "image", "image": png_data_uri(c)} for c in ((220, 20, 20), (20, 20, 220))]
        img_vecs = [d["embedding"] for d in client.post(
            "/v1/embeddings", json={"model": model, "input": images}).json()["data"]]
        cap_vecs = [d["embedding"] for d in embed(client, model, ["真っ赤な画像", "真っ青な画像"], "query")["data"]]
        assert _cosine(cap_vecs[0], img_vecs[0]) > _cosine(cap_vecs[0], img_vecs[1])
        assert _cosine(cap_vecs[1], img_vecs[1]) > _cosine(cap_vecs[1], img_vecs[0])


def _image_vector(client, model, image_uri, **extra):
    resp = client.post("/v1/embeddings", json={"model": model, "input": [{"type": "image", "image": image_uri}], **extra})
    assert resp.status_code == 200, resp.text[:300]
    return resp.json()["data"][0]["embedding"]


class TestImageTokens:
    """`image_tokens` sets the per-image token budget (BENCHMARK_REPORT.md §10)."""

    def test_omitted_equals_the_default(self, client, test_cases):
        case = test_cases["embedding_images"]
        page = text_page_data_uri()
        default = _image_vector(client, case["model"], page)
        explicit = _image_vector(client, case["model"], page, image_tokens=case["default_image_tokens"])
        assert _cosine(default, explicit) > 0.9999

    @pytest.mark.parametrize("tokens", [70, 560, 1120])
    def test_other_budgets_change_the_vector(self, client, test_cases, tokens):
        """The page is re-rendered at another resolution, so the vector moves but stays a unit vector."""
        case = test_cases["embedding_images"]
        assert tokens in case["image_token_options"]
        page = text_page_data_uri()
        default = _image_vector(client, case["model"], page)
        other = _image_vector(client, case["model"], page, image_tokens=tokens)
        assert len(other) == case["expected_dim"]
        assert 0.99 <= l2_norm(other) <= 1.01
        assert _cosine(default, other) < 0.9999

    def test_text_inputs_ignore_image_tokens(self, client, test_cases):
        model = test_cases["embedding_images"]["model"]
        plain = embed(client, model, ["画像トークン数はテキストに影響しない"])["data"][0]["embedding"]
        resp = client.post("/v1/embeddings", json={"model": model, "input": ["画像トークン数はテキストに影響しない"],
                                                   "image_tokens": 560})
        assert resp.status_code == 200, resp.text[:300]
        assert _cosine(plain, resp.json()["data"][0]["embedding"]) > 0.9999
