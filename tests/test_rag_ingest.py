import pytest

from app.rag.chunker import TextChunker
from app.rag.ingest import default_title, delete_document, ingest_document


class FakeCollection:
    def __init__(self):
        self.records: dict[str, dict] = {}

    def add(self, documents, ids, metadatas):
        assert len(documents) == len(ids) == len(metadatas)

        for text, identifier, metadata in zip(
            documents,
            ids,
            metadatas,
        ):
            self.records[identifier] = {
                "text": text,
                "metadata": metadata,
            }

    def get(self, where=None, include=None):
        document_id = (where or {}).get("doc_id")
        ids = [
            identifier
            for identifier, record in self.records.items()
            if record["metadata"].get("doc_id") == document_id
        ]

        return {
            "ids": ids,
            "documents": [
                self.records[identifier]["text"] for identifier in ids
            ],
            "metadatas": [
                self.records[identifier]["metadata"] for identifier in ids
            ],
        }

    def delete(self, ids=None, where=None):
        for identifier in ids or []:
            self.records.pop(identifier, None)


def test_ingest_stores_every_chunk_with_metadata():
    collection = FakeCollection()

    result = ingest_document(
        collection,
        "# 差旅报销标准\n一线城市每天500元。",
        permission="employee",
        uploaded_by=7,
    )

    assert result["doc_id"]
    assert result["title"] == "差旅报销标准"
    assert result["chunks"] == 1
    assert len(collection.records) == 1

    record = next(iter(collection.records.values()))
    metadata = record["metadata"]

    assert metadata["permission"] == "employee"
    assert metadata["title"] == "差旅报销标准"
    assert metadata["doc_id"] == result["doc_id"]
    assert metadata["chunk_index"] == 0
    assert metadata["source"] == "upload"
    assert metadata["uploaded_by"] == "7"
    assert metadata["created_at"]


def test_ingest_ids_are_scoped_to_document():
    collection = FakeCollection()

    result = ingest_document(
        collection,
        "第一段内容。" * 100,
        permission="public",
    )

    ids = sorted(collection.records)

    assert ids == [
        f"{result['doc_id']}:0",
        f"{result['doc_id']}:1",
    ]
    assert [
        record["metadata"]["chunk_index"]
        for record in collection.records.values()
    ] == [0, 1]


def test_ingest_uses_explicit_title():
    collection = FakeCollection()

    result = ingest_document(
        collection,
        "正文内容。",
        permission="public",
        title="  财务制度手册  ",
    )

    assert result["title"] == "财务制度手册"


def test_ingest_defaults_uploaded_by_to_empty_string():
    collection = FakeCollection()

    ingest_document(
        collection,
        "正文内容。",
        permission="public",
    )

    record = next(iter(collection.records.values()))

    assert record["metadata"]["uploaded_by"] == ""


def test_ingest_rejects_empty_text():
    collection = FakeCollection()

    with pytest.raises(ValueError):
        ingest_document(collection, "   ", permission="public")


def test_custom_chunker_is_used():
    collection = FakeCollection()

    result = ingest_document(
        collection,
        "abcdefghij",
        permission="public",
        chunker=TextChunker(chunk_size=4, overlap=0),
    )

    assert result["chunks"] == 3


def test_delete_document_removes_only_matching_document():
    collection = FakeCollection()
    first = ingest_document(collection, "第一份文档内容。", permission="public")
    second = ingest_document(collection, "第二份文档内容。", permission="public")

    removed = delete_document(collection, first["doc_id"])

    assert removed == 1
    assert [
        record["metadata"]["doc_id"]
        for record in collection.records.values()
    ] == [second["doc_id"]]


def test_delete_unknown_document_returns_zero():
    collection = FakeCollection()

    assert delete_document(collection, "missing") == 0


def test_default_title_falls_back_to_untitled():
    assert default_title("   \n  ") == "Untitled"
    assert default_title("# 制度手册", limit=2) == "制度"
