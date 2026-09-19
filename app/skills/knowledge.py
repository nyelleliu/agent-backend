from typing import Any

from app.skills.base import Skill


class KnowledgeSearchSkill(Skill):
    name = "knowledge_search"
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

    def __init__(self, collection):
        self.collection = collection

    def run(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query", "")

        if not query:
            return "Error: query is required"

        results = self.collection.query(
            query_texts=[query],
            n_results=3,
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
