"""Best-effort output-layer checks for suspicious reply content.

This is a monitoring aid, NOT a security boundary. It records a
structured log event when a final assistant reply appears to contain
credential-like material or artifacts of prompt-injection framing.
It never blocks, rewrites, or guarantees detection of anything; a
clean scan does not prove a reply is safe.

Detection reuses the credential regexes from app.core.errors so the
output and log layers share one definition of "looks like a secret".
"""

import logging
import re

from app.core.errors import (
    _BEARER,
    _JWT,
    _KEY_VALUE_SECRET,
    redact_secrets,
)

logger = logging.getLogger(__name__)


MAX_SNIPPET_CHARS = 80

# A URL whose authority section embeds credentials
# (scheme://user:pass@host). Ordinary URLs are not flagged.
_URL_WITH_CREDENTIALS = re.compile(r"://[^/\s:@]+:[^/\s@]+@")

# Structural artifacts: the model echoing the untrusted-document
# wrapper back out, or reproducing instruction-override phrasing.
# These are heuristics about the *shape* of the text, not a judgment
# about anyone's intent.
_UNTRUSTED_WRAPPER = re.compile(
    r"</?\s*untrusted_document\s*>",
    re.IGNORECASE,
)
_INSTRUCTION_OVERRIDE = re.compile(
    r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instructions?|rules?|prompts?)\b"
)


def _request_id():
    try:
        from app.core.logging_config import request_id_var

        return str(request_id_var.get() or "").strip()[:64] or "-"
    except Exception:
        return "-"


def scan_reply(reply) -> list[str]:
    """Return suspected-risk categories found in ``reply``.

    Categories:
    - ``suspected_secret_leak``: credential-like pattern (key=value
      secret, JWT, bearer token, or URL-embedded password).
    - ``suspected_injection_artifact``: structural echo of the
      untrusted-document wrapper or instruction-override phrasing.

    An empty list only means the narrow heuristics matched nothing.
    """
    if not reply or not isinstance(reply, str):
        return []

    categories = []

    secret_hit = (
        _KEY_VALUE_SECRET.search(reply)
        or _JWT.search(reply)
        or _BEARER.search(reply)
        or _URL_WITH_CREDENTIALS.search(reply)
    )

    if secret_hit:
        categories.append("suspected_secret_leak")

    if (
        _UNTRUSTED_WRAPPER.search(reply)
        or _INSTRUCTION_OVERRIDE.search(reply)
    ):
        categories.append("suspected_injection_artifact")

    return categories


def scan_and_log_reply(reply) -> list[str]:
    """Scan ``reply`` and log a warning when risks are suspected.

    Never raises. Logs only redacted, truncated match snippets —
    never the full reply and never the raw matched text.
    """
    try:
        categories = scan_reply(reply)

        if not categories:
            return []

        snippets = []

        for pattern in (
            _KEY_VALUE_SECRET,
            _JWT,
            _BEARER,
            _URL_WITH_CREDENTIALS,
            _UNTRUSTED_WRAPPER,
            _INSTRUCTION_OVERRIDE,
        ):
            match = pattern.search(reply or "")
            if match:
                snippets.append(match.group(0))

        sample = redact_secrets(" | ".join(snippets))
        sample = sample[:MAX_SNIPPET_CHARS]

        logger.warning(
            "output_risk_scan request_id=%s categories=%s "
            "sample=%s",
            _request_id(),
            ",".join(categories),
            sample,
        )

        return categories
    except Exception:
        logger.exception("output_risk_scan_failed")
        return []
