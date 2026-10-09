from app.skills.knowledge import KnowledgeSearchSkill


class FakeCollection:
    def __init__(self, documents=None):
        self.documents = (
            documents
            if documents is not None
            else [
                {
                    "id": "public-0",
                    "text": "public document",
                    "permission": "public",
                    "title": "Public Doc",
                },
                {
                    "id": "employee-0",
                    "text": "employee document",
                    "permission": "employee",
                    "title": "Employee Doc",
                },
                {
                    "id": "finance-0",
                    "text": "finance document",
                    "permission": "finance",
                    "title": "Finance Doc",
                },
                {
                    "id": "admin-0",
                    "text": "admin document",
                    "permission": "admin",
                    "title": "Admin Doc",
                },
            ]
        )
        self.queries = []

    def query(
        self,
        query_texts,
        n_results,
        where,
        include,
        where_document=None,
    ):
        self.queries.append(
            {
                "where": where,
                "where_document": where_document,
            }
        )

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
            "ids": [[document["id"] for document in rows]],
            "documents": [[document["text"] for document in rows]],
            "metadatas": [
                [
                    {
                        "permission": document["permission"],
                        "title": document.get("title", ""),
                    }
                    for document in rows
                ]
            ],
        }

        if "distances" in include:
            result["distances"] = [[0.5 for _ in rows]]

        return result


def _terms_from(where_document):
    if "$contains" in where_document:
        return [where_document["$contains"]]

    return [entry["$contains"] for entry in where_document["$or"]]


def create_skill(documents=None, top_k=10):
    collection = FakeCollection(documents)
    skill = KnowledgeSearchSkill(
        collection=collection,
        tool_registry=None,
        top_k=top_k,
    )
    return skill, collection


def used_permissions(collection):
    return [
        set(query["where"]["permission"]["$in"])
        for query in collection.queries
    ]


def test_employee_can_search_public_and_employee_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "employee",
    })

    assert "public document" in result
    assert "employee document" in result
    assert "finance document" not in result
    assert "admin document" not in result

    for permissions in used_permissions(collection):
        assert permissions == {"public", "employee"}


def test_finance_can_search_public_employee_and_finance_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "finance",
    })

    assert "public document" in result
    assert "employee document" in result
    assert "finance document" in result
    assert "admin document" not in result

    for permissions in used_permissions(collection):
        assert permissions == {"public", "employee", "finance"}


def test_admin_can_search_all_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "admin",
    })

    for text in (
        "public document",
        "employee document",
        "finance document",
        "admin document",
    ):
        assert text in result

    for permissions in used_permissions(collection):
        assert permissions == {"public", "employee", "finance", "admin"}


def test_unknown_role_can_only_search_public_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "unknown",
    })

    assert "public document" in result
    assert "employee document" not in result

    for permissions in used_permissions(collection):
        assert permissions == {"public"}


def test_every_query_is_permission_filtered():
    skill, _ = create_skill()

    skill.run({"query": "test", "_user_role": "finance"})

    queries = skill.retriever.collection.queries

    assert len(queries) == 2

    for query in queries:
        assert "permission" in query["where"]


def test_sources_are_returned_with_citations():
    skill, _ = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "employee",
    })

    assert result.startswith("[1] Public Doc (permission=public, score=")
    assert "[2] Employee Doc (permission=employee, score=" in result


def test_default_top_k_limits_the_number_of_sources():
    skill, _ = create_skill(top_k=3)

    result = skill.run({
        "query": "test",
        "_user_role": "admin",
    })

    assert result.count("(permission=") == 3
    assert "admin document" not in result


def test_missing_query_returns_error():
    skill, _ = create_skill()

    assert skill.run({"_user_role": "employee"}) == "Error: query is required"


def test_empty_knowledge_base_returns_message():
    skill, _ = create_skill(documents=[])

    result = skill.run({
        "query": "test",
        "_user_role": "employee",
    })

    assert result == "No relevant reference material found."
