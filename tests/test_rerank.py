"""POST /v1/rerank (and its /rerank alias)."""

import math

import pytest

from conftest import requires_vl, rerank

RERANK_MODELS = [
    "qwen3-0.6b",
    # pytest.param("qwen3-vl-reranker-2b", marks=[pytest.mark.vl, requires_vl]),
]


class TestRerankModels:
    @pytest.mark.parametrize("model_name", RERANK_MODELS)
    def test_ranking_and_response_shape(self, client, test_cases, model_name):
        case = test_cases["rerank"][model_name]
        data = rerank(client, model_name, case["query"], case["documents"], case["top_k"])

        assert set(data) == {"model", "results"}
        assert data["model"] == model_name

        results = data["results"]
        assert len(results) == min(case["top_k"], len(case["documents"]))

        for item in results:
            # `document` is popped by the server: only index + score come back.
            assert set(item) == {"index", "relevance_score"}
            assert 0 <= item["index"] < len(case["documents"])
            assert isinstance(item["relevance_score"], float)
            assert not math.isnan(item["relevance_score"])

        indexes = [item["index"] for item in results]
        assert len(set(indexes)) == len(indexes), "duplicate document index"

        scores = [item["relevance_score"] for item in results]
        assert scores == sorted(scores, reverse=True)

        assert results[0]["index"] == case["expected_first_index"]

    @pytest.mark.parametrize("model_name", RERANK_MODELS)
    def test_score_range(self, client, test_cases, model_name):
        """qwen3-0.6b returns a yes/no softmax probability, so 0 <= score <= 1."""
        case = test_cases["rerank"][model_name]
        if case["score_range"] is None:
            pytest.skip(f"{model_name} has no documented score range")
        low, high = case["score_range"]
        data = rerank(client, model_name, case["query"], case["documents"], len(case["documents"]))
        for item in data["results"]:
            assert low <= item["relevance_score"] <= high


class TestRerankRequestShapes:
    def test_default_model_when_model_omitted(self, client, test_cases):
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        data = rerank(client, None, case["query"], case["documents"])
        assert data["model"] == test_cases["rerank_default_model"]

    def test_alias_path_matches_v1_path(self, client, test_cases):
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        canonical = rerank(client, None, case["query"], case["documents"], path="/v1/rerank")
        alias = rerank(client, None, case["query"], case["documents"], path="/rerank")
        assert canonical == alias

    def test_top_k_limits_results(self, client, test_cases):
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        data = rerank(client, None, case["query"], case["documents"], top_k=1)
        assert len(data["results"]) == 1

    def test_top_k_zero_returns_every_document(self, client, test_cases):
        """`if req.top_k:` means 0 disables truncation."""
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        data = rerank(client, None, case["query"], case["documents"], top_k=0)
        assert len(data["results"]) == len(case["documents"])

    def test_top_k_larger_than_documents(self, client, test_cases):
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        data = rerank(client, None, case["query"], case["documents"], top_k=999)
        assert len(data["results"]) == len(case["documents"])

    def test_empty_documents_returns_empty_results(self, client):
        data = rerank(client, None, "クエリ", [])
        assert data["results"] == []

    def test_deterministic(self, client, test_cases):
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        first = rerank(client, None, case["query"], case["documents"], top_k=0)
        second = rerank(client, None, case["query"], case["documents"], top_k=0)
        assert first == second


class TestRerankInstruction:
    """qwen3-0.6b uses the model card's format, where `instruction` fills `<Instruct>`."""

    def test_omitted_equals_the_default_instruction(self, client, test_cases):
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        without = rerank(client, None, case["query"], case["documents"], top_k=0)
        explicit = rerank(client, None, case["query"], case["documents"], top_k=0,
                          instruction=case["default_instruction"])
        assert without["results"] == explicit["results"]

    def test_instruction_changes_the_scores(self, client, test_cases):
        case = test_cases["rerank"][test_cases["rerank_default_model"]]
        assert case["instruction_sensitive"] is True
        without = rerank(client, None, case["query"], case["documents"], top_k=0)
        with_instruction = rerank(client, None, case["query"], case["documents"], top_k=0,
                                  instruction="Judge whether the document is written in English")
        before = {r["index"]: r["relevance_score"] for r in without["results"]}
        after = {r["index"]: r["relevance_score"] for r in with_instruction["results"]}
        assert before != after


class TestRerankRetrievalSanity:
    """Prompt-format regression guard (BENCHMARK_REPORT.md §11).

    The server used to send a prompt without <|im_end|>, <Instruct> or the
    <think></think> block; on JaGovFAQs that ranked below the embeddings alone.
    Each query must put its own answer first, with a clear yes for it.
    """

    def test_each_query_ranks_its_answer_first(self, client, test_cases):
        case = test_cases["embedding_retrieval"]
        for i, query in enumerate(case["queries"]):
            results = rerank(client, None, query, case["documents"], top_k=0)["results"]
            assert results[0]["index"] == i, f"query {i}: top document {results[0]['index']}"
            assert results[0]["relevance_score"] > 0.5

    def test_long_documents_are_truncated_not_rejected(self, client, test_cases):
        """Documents are cut to RERANK_MAX_DOC_TOKENS, so a very long one still scores."""
        case = test_cases["embedding_retrieval"]
        long_doc = case["documents"][0] + "。" + "関係のない文章が続きます。" * 2000
        results = rerank(client, None, case["queries"][0], [long_doc, case["documents"][1]], top_k=0)["results"]
        assert len(results) == 2
        assert all(0.0 <= r["relevance_score"] <= 1.0 for r in results)
