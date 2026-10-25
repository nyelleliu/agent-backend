import json
from typing import Any


DEFAULT_MAX_CONTEXT_CHARS = 24_000
MIN_MAX_CONTEXT_CHARS = 1_000


def effective_context_budget(max_context_chars: int) -> int:
    """Apply the shared lower bound used by Planner and AgentLoop."""
    return max(MIN_MAX_CONTEXT_CHARS, max_context_chars)


class ContextBudgetExceeded(ValueError):
    """A model request exceeds its configured serialized-size budget."""


def serialized_request_size(
    model: str,
    messages: list[dict[str, Any]],
    **request_fields: Any,
) -> int:
    """Count the compact JSON request body, including all supplied fields."""
    request_payload = {
        "model": model,
        "messages": messages,
        **request_fields,
    }
    try:
        serialized = json.dumps(
            request_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("request payload is not valid JSON") from exc
    return len(serialized)
