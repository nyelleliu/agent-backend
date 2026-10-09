from types import SimpleNamespace

from app.rag.reranker import HeuristicReranker, LLMReranker
from app.rag.retriever import RetrievedChunk


def make_chunk(identifier, text, score=0.5, title=""):
    return RetrievedChunk(
        id=identifier,
        text=text,
        score=score,
        metadata={"title": title, "permission": "public"},
        matched_by=("vector",),
    )


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )


def make_client(content=None, error=None):
    completions = FakeCompletions(content=content, error=error)

    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    ), completions


def test_query_terms_pull_relevant_passage_forward():
    irrelevant = make_chunk(
        "irrelevant",
        "市内交通费每天上限是100元。",
        score=0.95,
    )
    relevant = make_chunk(
        "relevant",
        "差旅报销标准是一线城市每天500元。",
        score=0.55,
    )

    ranked = HeuristicReranker().rerank(
        "差旅报销标准是多少",
        [irrelevant, relevant],
        top_k=2,
    )

    assert [chunk.id for chunk in ranked] == ["relevant", "irrelevant"]


def test_title_match_boosts_relevant_passage():
    without_title = make_chunk(
        "plain",
        "一线城市每天500元。",
        score=0.6,
        title="其他文档",
    )
    with_title = make_chunk(
        "titled",
        "一线城市每天500元。",
        score=0.6,
        title="差旅报销标准",
    )

    ranked = HeuristicReranker().rerank(
        "差旅报销标准",
        [without_title, with_title],
        top_k=2,
    )

    assert [chunk.id for chunk in ranked] == ["titled", "plain"]


def test_without_query_terms_retrieval_order_is_kept():
    first = make_chunk("first", "aaaa", score=0.9)
    second = make_chunk("second", "bbbb", score=0.4)

    ranked = HeuristicReranker().rerank(
        "的了",
        [first, second],
        top_k=2,
    )

    assert [chunk.id for chunk in ranked] == ["first", "second"]


def test_top_k_is_applied():
    chunks = [
        make_chunk(f"doc-{index}", "差旅报销标准", score=0.5)
        for index in range(4)
    ]

    ranked = HeuristicReranker().rerank(
        "差旅报销",
        chunks,
        top_k=2,
    )

    assert len(ranked) == 2


def test_empty_input_returns_empty_list():
    assert HeuristicReranker().rerank("查询", [], top_k=3) == []


def test_llm_reranker_uses_model_order():
    chunks = [
        make_chunk("a", "aaa", score=0.9),
        make_chunk("b", "bbb", score=0.1),
    ]
    client, completions = make_client(
        content='{"ranked_ids": ["b", "a"]}'
    )

    ranked = LLMReranker(client).rerank("查询", chunks, top_k=2)

    assert [chunk.id for chunk in ranked] == ["b", "a"]
    assert completions.calls[0]["temperature"] == 0


def test_llm_reranker_keeps_unknown_ids_out():
    chunks = [make_chunk("a", "aaa"), make_chunk("b", "bbb")]
    client, _ = make_client(content='{"ranked_ids": ["b", "ghost"]}')

    ranked = LLMReranker(client).rerank("查询", chunks, top_k=2)

    assert [chunk.id for chunk in ranked] == ["b", "a"]


def test_llm_reranker_falls_back_on_error():
    chunks = [
        make_chunk("weak", "无关内容", score=0.95),
        make_chunk("strong", "差旅报销标准", score=0.4),
    ]
    client, _ = make_client(error=RuntimeError("boom"))

    ranked = LLMReranker(client).rerank(
        "差旅报销标准",
        chunks,
        top_k=2,
    )

    assert [chunk.id for chunk in ranked] == ["strong", "weak"]
