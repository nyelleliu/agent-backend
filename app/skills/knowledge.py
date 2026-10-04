from typing import Any

from app.permissions.checker import PermissionChecker
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

    def __init__(self, collection, tool_registry):
        self.collection = collection
        self.tool_registry = tool_registry
        self.permission_checker = PermissionChecker()

    def run(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query", "")
        role = arguments.get("_user_role", "employee")

        if not query:
            return "Error: query is required"

        allowed_permissions = self.permission_checker.DOCUMENT_PERMISSIONS.get(
            role,
            {"public"},
        )

        results = self.collection.query(
            query_texts=[query],
            n_results=3,
            where={
                "permission": {
                    "$in": list(allowed_permissions)
                }
            },
            include=["documents", "distances"],
        )

        distances = results["distances"][0]
        documents = results["documents"][0]

        filtered_chunks = []

        for i in range(len(documents)):
            if distances[i] < 1.5:
                filtered_chunks.append(documents[i])

        if not filtered_chunks:
            return "No relevant reference material found."

        return "\n".join(filtered_chunks)
