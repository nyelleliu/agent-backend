import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.permissions.checker import PermissionChecker
from app.rag.retriever import RetrievedChunk


Runner = Callable[["EvalCase"], list[RetrievedChunk]]


@dataclass
class EvalCase:
    query: str
    role: str
    relevant_titles: list[str] = field(default_factory=list)
    forbidden_titles: list[str] = field(default_factory=list)
    name: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalCase":
        if not isinstance(payload, dict):
            raise ValueError("each case must be an object")

        query = payload.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("each case needs a non-empty query")

        role = payload.get("role")

        if not isinstance(role, str) or not role:
            raise ValueError("each case needs a role")

        relevant = payload.get("relevant_titles", [])
        forbidden = payload.get("forbidden_titles", [])

        if not isinstance(relevant, list) or not all(
            isinstance(value, str) for value in relevant
        ):
            raise ValueError("relevant_titles must be a list of strings")

        if not isinstance(forbidden, list) or not all(
            isinstance(value, str) for value in forbidden
        ):
            raise ValueError("forbidden_titles must be a list of strings")

        name = payload.get("name", "")

        if not isinstance(name, str):
            raise ValueError("name must be a string")

        return cls(
            query=query,
            role=role,
            relevant_titles=relevant,
            forbidden_titles=forbidden,
            name=name,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "query": self.query,
            "role": self.role,
            "relevant_titles": list(self.relevant_titles),
            "forbidden_titles": list(self.forbidden_titles),
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = []

    for line_number, payload in enumerate(
        load_jsonl(path),
        start=1,
    ):
        try:
            cases.append(EvalCase.from_dict(payload))
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: {exc}"
            ) from exc

    if not cases:
        raise ValueError(f"{path}: no evaluation cases found")

    return cases


def load_jsonl(path: str | Path) -> list[dict]:
    file_path = Path(path)

    if not file_path.exists():
        raise ValueError(f"{file_path} does not exist")

    payloads = []

    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            payloads.append(json.loads(line))

    return payloads


def evaluate(
    runner: Runner,
    cases: list[EvalCase],
    top_k: int = 3,
    permission_checker: PermissionChecker | None = None,
) -> dict:
    checker = permission_checker or PermissionChecker()
    results = []

    for case in cases:
        chunks = runner(case)[:top_k]
        titles = [
            (chunk.metadata or {}).get("title") or ""
            for chunk in chunks
        ]
        leaks = _find_leaks(case, chunks, checker)
        recall, rank = _relevance(case, titles)

        results.append(
            {
                "name": case.name,
                "query": case.query,
                "role": case.role,
                "retrieved_titles": titles,
                "relevant_titles": list(case.relevant_titles),
                "recall": recall,
                "rank": rank,
                "leaks": leaks,
            }
        )

    scored = [result for result in results if result["relevant_titles"]]

    hits = [result for result in scored if result["rank"] is not None]
    total = len(scored)

    return {
        "cases": len(cases),
        "scored_cases": total,
        "top_k": top_k,
        "hit_rate": round(len(hits) / total, 4) if total else 0.0,
        "recall_at_k": round(
            sum(result["recall"] for result in scored) / total,
            4,
        )
        if total
        else 0.0,
        "mrr": round(
            sum(1 / result["rank"] for result in hits) / total,
            4,
        )
        if total
        else 0.0,
        "no_result_rate": round(
            sum(1 for result in results if not result["retrieved_titles"])
            / len(results),
            4,
        )
        if results
        else 0.0,
        "permission_leaks": sum(
            len(result["leaks"]) for result in results
        ),
        "results": results,
    }


def _relevance(case: EvalCase, titles: list[str]) -> tuple[float, int | None]:
    expected = set(case.relevant_titles)

    if not expected:
        return 0.0, None

    retrieved = [
        title
        for title in titles
        if title in expected
    ]

    if not retrieved:
        return 0.0, None

    rank = titles.index(retrieved[0]) + 1
    recall = len(set(retrieved)) / len(expected)

    return round(recall, 4), rank


def _find_leaks(
    case: EvalCase,
    chunks: list[RetrievedChunk],
    checker: PermissionChecker,
) -> list[str]:
    allowed = checker.DOCUMENT_PERMISSIONS.get(case.role, {"public"})
    leaks = []

    for chunk in chunks:
        metadata = chunk.metadata or {}
        title = metadata.get("title") or ""
        permission = metadata.get("permission")

        if permission not in allowed:
            leaks.append(f"{title or chunk.id} permission={permission}")

        if title and title in case.forbidden_titles:
            leaks.append(f"{title} forbidden_for_role={case.role}")

    return leaks
