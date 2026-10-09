from app.rag.retriever import Retriever


class FakeCollection:
    def __init__(self, documents, distances=None):
        self.documents = documents
        self.distances = distances or {}
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
                "query": query_texts[0],
                "where": where,
                "where_document": where_document,
                "include": include,
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
            result["distances"] = [
                [
                    self.distances.get(document["id"], 0.5)
                    for document in rows
                ]
            ]

        return result


def _terms_from(where_document):
    if "$contains" in where_document:
        return [where_document["$contains"]]

    return [entry["$contains"] for entry in where_document["$or"]]


def build_collection():
    documents = [
        {
            "id": "public-0",
            "text": "差旅报销标准是一线城市每天500元。",
            "permission": "public",
            "title": "差旅报销标准",
        },
        {
            "id": "employee-0",
            "text": "员工手册规定了考勤和请假流程。",
            "permission": "employee",
            "title": "员工手册",
        },
        {
            "id": "finance-0",
            "text": "财务制度说明了发票报销的审批流程。",
            "permission": "finance",
            "title": "财务制度",
        },
        {
            "id": "admin-0",
            "text": "管理员手册描述了系统权限配置。",
            "permission": "admin",
            "title": "管理员手册",
        },
    ]
    distances = {
        "public-0": 0.4,
        "employee-0": 0.9,
        "finance-0": 0.2,
        "admin-0": 0.1,
    }

    return FakeCollection(documents, distances)


def test_results_are_limited_to_allowed_permissions():
    collection = build_collection()
    retriever = Retriever(collection)

    results = retriever.retrieve(
        "差旅报销",
        {"public", "employee"},
        top_k=10,
    )

    permissions = {
        chunk.metadata["permission"] for chunk in results
    }

    assert permissions <= {"public", "employee"}

    for query in collection.queries:
        assert set(query["where"]["permission"]["$in"]) == {
            "public",
            "employee",
        }


def test_vector_results_above_distance_threshold_are_dropped():
    documents = [
        {
            "id": "near",
            "text": "差旅报销标准",
            "permission": "public",
            "title": "近",
        },
        {
            "id": "far",
            "text": "完全无关的一段内容",
            "permission": "public",
            "title": "远",
        },
    ]
    collection = FakeCollection(
        documents,
        {"near": 0.4, "far": 3.0},
    )

    results = Retriever(collection).retrieve(
        "差旅报销",
        {"public"},
        top_k=10,
    )

    assert [chunk.id for chunk in results] == ["near"]


def test_keyword_leg_recovers_relevant_document():
    documents = [
        {
            "id": "far-match",
            "text": "差旅报销标准是一线城市每天500元",
            "permission": "public",
            "title": "差旅",
        },
        {
            "id": "near",
            "text": "无关内容",
            "permission": "public",
            "title": "无关",
        },
    ]
    collection = FakeCollection(
        documents,
        {"far-match": 3.0, "near": 0.6},
    )

    results = Retriever(collection).retrieve(
        "差旅报销",
        {"public"},
        top_k=10,
    )
    by_id = {chunk.id: chunk for chunk in results}

    assert "far-match" in by_id
    assert by_id["far-match"].matched_by == ("keyword",)
    assert by_id["far-match"].score > 0


def test_document_matched_by_both_legs_is_merged_once():
    documents = [
        {
            "id": "both",
            "text": "差旅报销标准是一线城市每天500元",
            "permission": "public",
            "title": "差旅",
        }
    ]
    collection = FakeCollection(documents, {"both": 0.4})

    results = Retriever(collection).retrieve(
        "差旅报销",
        {"public"},
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].matched_by == ("vector", "keyword")
    assert results[0].score > 1 / (1 + 0.4)
    assert len(collection.queries) == 2


def test_better_distance_ranks_higher():
    documents = [
        {
            "id": "worse",
            "text": "alpha content",
            "permission": "public",
            "title": "worse",
        },
        {
            "id": "better",
            "text": "alpha content",
            "permission": "public",
            "title": "better",
        },
    ]
    collection = FakeCollection(
        documents,
        {"worse": 1.2, "better": 0.1},
    )

    results = Retriever(collection).retrieve(
        "alpha",
        {"public"},
        top_k=10,
    )

    assert [chunk.id for chunk in results] == ["better", "worse"]


def test_top_k_limits_the_result_count():
    documents = [
        {
            "id": f"doc-{index}",
            "text": "差旅报销标准",
            "permission": "public",
            "title": f"doc-{index}",
        }
        for index in range(5)
    ]
    collection = FakeCollection(documents)

    results = Retriever(collection).retrieve(
        "差旅报销",
        {"public"},
        top_k=2,
    )

    assert len(results) == 2


def test_empty_query_returns_nothing():
    collection = build_collection()

    results = Retriever(collection).retrieve("   ", {"public"})

    assert results == []
    assert collection.queries == []


def test_empty_permissions_return_nothing():
    collection = build_collection()

    results = Retriever(collection).retrieve("差旅报销", set())

    assert results == []
    assert collection.queries == []
