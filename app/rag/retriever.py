import hashlib
import logging
from dataclasses import dataclass, field

from app.rag.terms import extract_terms


logger = logging.getLogger(__name__)


DEFAULT_N_RESULTS = 10
DEFAULT_DISTANCE_THRESHOLD = 1.5
DEFAULT_KEYWORD_LIMIT = 8
KEYWORD_SCORE_WEIGHT = 0.8
MULTI_MATCH_BOOST = 0.05


def _valid_candidate(text, metadata, allowed_permissions) -> bool:
    """Minimum write-contract check shared by both retrieval legs.

    Records that are incomplete (legacy imports without doc_id /
    title / chunk_index), malformed, or carrying a permission outside
    the request's allowed set are rejected outright — never repaired
    or faked into looking like valid documents. The Chroma ``where``
    filter is treated as an optimisation, not as the only enforcement.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    if not isinstance(metadata, dict):
        return False

    permission = metadata.get("permission")

    if not isinstance(permission, str) or not permission:
        return False

    if permission not in allowed_permissions:
        return False

    doc_id = metadata.get("doc_id")

    if not isinstance(doc_id, str) or not doc_id:
        return False

    title = metadata.get("title")

    if not isinstance(title, str) or not title.strip():
        return False

    chunk_index = metadata.get("chunk_index")

    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
    ):
        return False

    return True


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    matched_by: tuple[str, ...] = ()


class Retriever:
    def __init__(
        self,
        collection,
        n_results: int = DEFAULT_N_RESULTS,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        keyword_limit: int = DEFAULT_KEYWORD_LIMIT,
    ):
        self.collection = collection
        self.n_results = n_results
        self.distance_threshold = distance_threshold
        self.keyword_limit = keyword_limit

    def retrieve(
        self,
        query: str,
        allowed_permissions,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        query = (query or "").strip()
        permissions = sorted(allowed_permissions or [])

        if not query or not permissions:
            return []

        candidates: dict[str, RetrievedChunk] = {}

        self._collect(
            self._vector_query(query, permissions),
            candidates,
            source="vector",
        )
        self._collect(
            self._keyword_query(query, permissions),
            candidates,
            source="keyword",
        )

        ranked = sorted(
            candidates.values(),
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return ranked[:top_k]

    def _permission_filter(self, permissions: list[str]) -> dict:
        return {"permission": {"$in": permissions}}

    def _vector_query(self, query: str, permissions: list[str]) -> list[dict]:
        try:
            result = self.collection.query(
                query_texts=[query],
                n_results=self.n_results,
                where=self._permission_filter(permissions),
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:
            logger.warning("vector_query_failed error=%s", exc)
            return []

        documents = (result or {}).get("documents") or [[]]
        distances = (result or {}).get("distances") or [[]]
        metadatas = (result or {}).get("metadatas") or [[]]
        ids = (result or {}).get("ids") or [[]]

        rows = []

        for index, document in enumerate(documents[0] if documents else []):
            if not document:
                continue

            distance = distances[0][index] if distances[0] else None

            if distance is not None and distance >= self.distance_threshold:
                continue

            metadata = metadatas[0][index] if metadatas[0] else None

            if not _valid_candidate(document, metadata, permissions):
                continue

            rows.append(
                {
                    "id": self._row_id(ids, index, document),
                    "text": document,
                    "metadata": metadata,
                    "score": (
                        1.0 / (1.0 + distance)
                        if distance is not None
                        else 0.0
                    ),
                }
            )

        return rows

    def _keyword_query(self, query: str, permissions: list[str]) -> list[dict]:
        terms = extract_terms(query, limit=self.keyword_limit)

        if not terms:
            return []

        if len(terms) == 1:
            where_document = {"$contains": terms[0]}
        else:
            where_document = {
                "$or": [{"$contains": term} for term in terms]
            }

        try:
            result = self.collection.query(
                query_texts=[query],
                n_results=self.n_results,
                where=self._permission_filter(permissions),
                where_document=where_document,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("keyword_query_failed error=%s", exc)
            return []

        documents = (result or {}).get("documents") or [[]]
        metadatas = (result or {}).get("metadatas") or [[]]
        ids = (result or {}).get("ids") or [[]]

        rows = []

        for index, document in enumerate(documents[0] if documents else []):
            if not document:
                continue

            metadata = metadatas[0][index] if metadatas[0] else None

            if not _valid_candidate(document, metadata, permissions):
                continue

            lowered = document.lower()
            hits = sum(1 for term in terms if term in lowered)

            if not hits:
                continue

            rows.append(
                {
                    "id": self._row_id(ids, index, document),
                    "text": document,
                    "metadata": metadata,
                    "score": KEYWORD_SCORE_WEIGHT * hits / len(terms),
                }
            )

        return rows

    @staticmethod
    def _row_id(ids, index: int, text: str) -> str:
        if ids and ids[0] and index < len(ids[0]) and ids[0][index]:
            return ids[0][index]

        digest = hashlib.md5(text.encode("utf-8")).hexdigest()

        return f"text:{digest}"

    @staticmethod
    def _collect(rows: list[dict], candidates: dict, source: str):
        for row in rows:
            existing = candidates.get(row["id"])

            if existing is None:
                candidates[row["id"]] = RetrievedChunk(
                    id=row["id"],
                    text=row["text"],
                    score=row["score"],
                    metadata=row["metadata"],
                    matched_by=(source,),
                )
                continue

            if source in existing.matched_by:
                existing.score = max(existing.score, row["score"])
                continue

            existing.matched_by = (*existing.matched_by, source)
            existing.score = min(
                1.0,
                max(existing.score, row["score"]) + MULTI_MATCH_BOOST,
            )
