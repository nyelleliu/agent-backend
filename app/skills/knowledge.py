from typing import Any

from app.permissions.checker import PermissionChecker
from app.rag.reranker import HeuristicReranker, Reranker
from app.rag.retriever import Retriever
from app.skills.base import Skill


class KnowledgeSearchSkill(Skill):
    name = "knowledge_search"
    permission = "knowledge_search"
    description = "Search the enterprise knowledge base for relevant information"

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question or search query",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        collection,
        tool_registry,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        permission_checker: PermissionChecker | None = None,
        candidate_k: int = 10,
        top_k: int = 3,
    ):
        self.collection = collection
        self.tool_registry = tool_registry
        self.permission_checker = permission_checker or PermissionChecker()
        self.retriever = retriever or Retriever(
            collection,
            n_results=candidate_k,
        )
        self.reranker = reranker or HeuristicReranker()
        self.candidate_k = candidate_k
        self.top_k = top_k

    def run(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query", "")
        role = arguments.get("_user_role")

        if not query:
            return "Error: query is required"

        ranked = self.search(query, role)

        if not ranked:
            return "No relevant reference material found."

        return self._format(ranked)

    def search(self, query: str, role: str | None = None) -> list:
        allowed_permissions = self.permission_checker.DOCUMENT_PERMISSIONS.get(
            role,
            {"public"},
        )

        candidates = self.retriever.retrieve(
            query,
            allowed_permissions,
            top_k=self.candidate_k,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=self.top_k,
        )

    @staticmethod
    def _format(chunks) -> str:
        blocks = []

        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata or {}
            title = metadata.get("title") or "Untitled"
            permission = metadata.get("permission", "public")

            blocks.append(
                f"[{index}] {title} "
                f"(permission={permission}, score={chunk.score:.2f})\n"
                f"{chunk.text.strip()}"
            )

        return "\n\n".join(blocks)
