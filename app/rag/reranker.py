import json
import logging
from abc import ABC, abstractmethod

from app.rag.retriever import RetrievedChunk
from app.rag.terms import extract_terms


logger = logging.getLogger(__name__)


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class HeuristicReranker(Reranker):
    def __init__(
        self,
        text_weight: float = 0.45,
        title_weight: float = 0.3,
        retrieval_weight: float = 0.25,
    ):
        self.text_weight = text_weight
        self.title_weight = title_weight
        self.retrieval_weight = retrieval_weight

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        query_terms = set(extract_terms(query))
        scored = []

        for chunk in chunks:
            if not query_terms:
                relevance = min(chunk.score, 1.0)
            else:
                text_terms = set(extract_terms(chunk.text))
                title_terms = set(
                    extract_terms(chunk.metadata.get("title") or "")
                )

                text_coverage = (
                    len(query_terms & text_terms) / len(query_terms)
                )
                title_coverage = (
                    len(query_terms & title_terms) / len(query_terms)
                    if title_terms
                    else 0.0
                )

                relevance = (
                    self.text_weight * text_coverage
                    + self.title_weight * title_coverage
                    + self.retrieval_weight * min(chunk.score, 1.0)
                )

            scored.append((relevance, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)

        return [chunk for _, chunk in scored[:top_k]]


class LLMReranker(Reranker):
    def __init__(
        self,
        client,
        model: str = "deepseek-chat",
        fallback: Reranker | None = None,
        max_candidates: int = 10,
    ):
        self.client = client
        self.model = model
        self.fallback = fallback or HeuristicReranker()
        self.max_candidates = max_candidates

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        candidates = list(chunks)[: self.max_candidates]

        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates[:top_k]

        prompt = self._build_prompt(query, candidates)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rank candidate passages by how well they "
                            "answer the user's question. "
                            "Respond with JSON: "
                            '{"ranked_ids": ["id1", "id2"]}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            order = json.loads(content).get("ranked_ids", [])
        except Exception as exc:
            logger.warning("llm_rerank_failed error=%s", exc)
            return self.fallback.rerank(query, chunks, top_k)

        by_id = {chunk.id: chunk for chunk in candidates}
        selected_ids = [value for value in order if value in by_id]
        ranked = [by_id[value] for value in selected_ids]
        ranked_ids = set(selected_ids)

        ranked.extend(
            chunk for chunk in candidates if chunk.id not in ranked_ids
        )

        return ranked[:top_k]

    @staticmethod
    def _build_prompt(query: str, candidates: list[RetrievedChunk]) -> str:
        entries = []

        for chunk in candidates:
            metadata = chunk.metadata or {}

            entries.append(
                {
                    "id": chunk.id,
                    "title": metadata.get("title", ""),
                    "text": chunk.text[:500],
                }
            )

        return json.dumps(
            {
                "question": query,
                "candidates": entries,
            },
            ensure_ascii=False,
        )
