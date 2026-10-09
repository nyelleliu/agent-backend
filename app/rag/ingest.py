import uuid
from datetime import datetime, timezone

from app.rag.chunker import TextChunker


def default_title(text: str, limit: int = 60) -> str:
    for line in (text or "").splitlines():
        title = line.strip().lstrip("#").strip()

        if title:
            return title[:limit]

    return "Untitled"


def ingest_document(
    collection,
    text: str,
    *,
    permission: str,
    title: str | None = None,
    uploaded_by=None,
    source: str = "upload",
    chunker: TextChunker | None = None,
) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("document text is empty")

    chunker = chunker or TextChunker()
    chunks = chunker.split(text)

    if not chunks:
        raise ValueError("document text is empty")

    document_id = uuid.uuid4().hex
    document_title = (title or "").strip() or default_title(text)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(f"{document_id}:{chunk.index}")
        documents.append(chunk.text)
        metadatas.append(
            {
                "permission": permission,
                "title": document_title,
                "doc_id": document_id,
                "chunk_index": chunk.index,
                "source": source,
                "uploaded_by": "" if uploaded_by is None else str(uploaded_by),
                "created_at": created_at,
            }
        )

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    return {
        "doc_id": document_id,
        "title": document_title,
        "chunks": len(chunks),
    }


def delete_document(collection, document_id: str) -> int:
    existing = collection.get(where={"doc_id": document_id})
    ids = existing.get("ids") or []

    if not ids:
        return 0

    collection.delete(ids=ids)

    return len(ids)
