import os

import pytest

from app.rag.evaluation import (
    EvalCase,
    evaluate,
    load_cases,
    load_jsonl,
)
from app.rag.retriever import RetrievedChunk
from app.skills.knowledge import KnowledgeSearchSkill


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_PATH = os.path.join(BASE_DIR, "evals", "rag_cases.jsonl")
CORPUS_PATH = os.path.join(BASE_DIR, "evals", "corpus.jsonl")


def chunk(title, permission="public", score=0.5):
    return RetrievedChunk(
        id=f"{title}:0",
        text=f"{title} content",
        score=score,
        metadata={"title": title, "permission": permission},
    )


def case(**kwargs):
    payload = {
        "query": "查询",
        "role": "employee",
    }
    payload.update(kwargs)

    return EvalCase.from_dict(payload)


def test_dataset_file_is_valid():
    cases = load_cases(CASES_PATH)
    corpus = load_jsonl(CORPUS_PATH)
    titles = {document["title"] for document in corpus}
    roles = {entry.role for entry in cases}

    assert len(cases) >= 6
    assert roles <= {"employee", "finance", "admin"}

    for entry in cases:
        assert set(entry.relevant_titles) <= titles
        assert set(entry.forbidden_titles) <= titles

    assert any(not entry.relevant_titles for entry in cases)


def test_load_cases_rejects_missing_file():
    with pytest.raises(ValueError, match="does not exist"):
        load_cases(os.path.join(BASE_DIR, "evals", "missing.jsonl"))


def test_load_cases_reports_line_number_for_bad_row(tmp_path):
    path = tmp_path / "bad_cases.jsonl"
    path.write_text(
        '{"query": "ok", "role": "employee"}\n'
        '{"query": "", "role": "employee"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bad_cases.jsonl:2"):
        load_cases(path)


def test_evaluate_perfect_ranking():
    runner = lambda entry: [chunk("A"), chunk("B")]
    report = evaluate(
        runner,
        [case(relevant_titles=["A"])],
        top_k=3,
    )

    assert report["hit_rate"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["permission_leaks"] == 0
    assert report["results"][0]["rank"] == 1


def test_evaluate_second_rank_lowers_mrr():
    runner = lambda entry: [chunk("A"), chunk("B")]
    report = evaluate(
        runner,
        [case(relevant_titles=["B"])],
        top_k=3,
    )

    assert report["hit_rate"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 0.5
    assert report["results"][0]["rank"] == 2


def test_evaluate_miss_counts_as_zero():
    runner = lambda entry: [chunk("A"), chunk("B")]
    report = evaluate(
        runner,
        [case(relevant_titles=["C"])],
        top_k=3,
    )

    assert report["hit_rate"] == 0.0
    assert report["recall_at_k"] == 0.0
    assert report["mrr"] == 0.0
    assert report["results"][0]["rank"] is None


def test_evaluate_aggregates_multiple_cases():
    runners = {
        "first": lambda entry: [chunk("A")],
        "second": lambda entry: [chunk("A"), chunk("B")],
    }
    cases = [
        case(query="first", relevant_titles=["A"]),
        case(query="second", relevant_titles=["B"]),
    ]
    report = evaluate(lambda entry: runners[entry.query](entry), cases)

    assert report["cases"] == 2
    assert report["scored_cases"] == 2
    assert report["hit_rate"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 0.75


def test_evaluate_detects_permission_leak():
    runner = lambda entry: [
        chunk("财务制度", permission="finance"),
        chunk("员工手册", permission="employee"),
    ]
    report = evaluate(runner, [case()], top_k=3)

    assert report["permission_leaks"] == 1
    assert report["results"][0]["leaks"] == [
        "财务制度 permission=finance"
    ]


def test_evaluate_detects_forbidden_title():
    runner = lambda entry: [chunk("系统管理员手册", permission="employee")]
    entry = case(forbidden_titles=["系统管理员手册"])

    report = evaluate(lambda case_: runner(case_), [entry])

    assert report["permission_leaks"] == 1
    assert "forbidden_for_role=employee" in report["results"][0]["leaks"][0]


def test_evaluate_ignores_negative_cases_in_scoring():
    runner = lambda entry: [chunk("公司简介", permission="public")]
    report = evaluate(
        runner,
        [case(relevant_titles=[], forbidden_titles=[])],
        top_k=3,
    )

    assert report["scored_cases"] == 0
    assert report["hit_rate"] == 0.0
    assert report["mrr"] == 0.0
    assert report["no_result_rate"] == 0.0
    assert report["permission_leaks"] == 0


class FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def query(
        self,
        query_texts,
        n_results,
        where,
        include,
        where_document=None,
    ):
        allowed = set(where["permission"]["$in"])
        rows = [
            document
            for document in self.documents
            if document["permission"] in allowed
        ]

        if where_document is not None:
            terms = _terms_from(where_document)
            rows = [
                document
                for document in rows
                if any(term in document["text"] for term in terms)
            ]

        rows = rows[:n_results]

        result = {
            "ids": [[document["title"] for document in rows]],
            "documents": [[document["text"] for document in rows]],
            "metadatas": [
                [
                    {
                        "title": document["title"],
                        "permission": document["permission"],
                    }
                    for document in rows
                ]
            ],
        }

        if "distances" in include:
            result["distances"] = [[0.6 for _ in rows]]

        return result


def _terms_from(where_document):
    if "$contains" in where_document:
        return [where_document["$contains"]]

    return [entry["$contains"] for entry in where_document["$or"]]


def test_evaluation_runs_against_the_dataset():
    corpus = load_jsonl(CORPUS_PATH)
    collection = FakeCollection(corpus)
    skill = KnowledgeSearchSkill(collection, None, top_k=3)
    cases = load_cases(CASES_PATH)

    report = evaluate(
        lambda entry: skill.search(entry.query, entry.role),
        cases,
        top_k=3,
    )

    assert report["cases"] == len(cases)
    assert report["permission_leaks"] == 0
    assert report["hit_rate"] >= 0.75
    assert report["mrr"] > 0

    for result in report["results"]:
        for title in result["relevant_titles"]:
            if result["rank"] is not None:
                assert title in result["retrieved_titles"]
