from app.skills.knowledge import KnowledgeSearchSkill


class FakeCollection:
    def __init__(self):
        self.last_query = None

        self.documents = [
            {
                "text": "public document",
                "permission": "public",
            },
            {
                "text": "employee document",
                "permission": "employee",
            },
            {
                "text": "finance document",
                "permission": "finance",
            },
            {
                "text": "admin document",
                "permission": "admin",
            },
        ]

    def query(self, **kwargs):
        self.last_query = kwargs

        allowed_permissions = set(
            kwargs["where"]["permission"]["$in"]
        )

        matched = [
            document
            for document in self.documents
            if document["permission"] in allowed_permissions
        ]

        return {
            "distances": [
                [0.5 for _ in matched]
            ],
            "documents": [
                [document["text"] for document in matched]
            ],
        }


def create_skill():
    collection = FakeCollection()
    skill = KnowledgeSearchSkill(
        collection=collection,
        tool_registry=None,
    )
    return skill, collection


def test_employee_can_search_public_and_employee_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "employee",
    })

    assert result == "public document\nemployee document"

    permissions = set(
        collection.last_query["where"]["permission"]["$in"]
    )

    assert permissions == {
        "public",
        "employee",
    }


def test_finance_can_search_public_employee_and_finance_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "finance",
    })

    assert result == (
        "public document\n"
        "employee document\n"
        "finance document"
    )

    permissions = set(
        collection.last_query["where"]["permission"]["$in"]
    )

    assert permissions == {
        "public",
        "employee",
        "finance",
    }


def test_admin_can_search_all_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "admin",
    })

    assert result == (
        "public document\n"
        "employee document\n"
        "finance document\n"
        "admin document"
    )

    permissions = set(
        collection.last_query["where"]["permission"]["$in"]
    )

    assert permissions == {
        "public",
        "employee",
        "finance",
        "admin",
    }


def test_unknown_role_can_only_search_public_documents():
    skill, collection = create_skill()

    result = skill.run({
        "query": "test",
        "_user_role": "unknown",
    })

    assert result == "public document"

    permissions = set(
        collection.last_query["where"]["permission"]["$in"]
    )

    assert permissions == {
        "public",
    }
