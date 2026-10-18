import logging
import os
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_HANDLER_MARKER = "_agent_backend_handler"


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging() -> None:
    """Configure application-wide logging.

    Logs go to the console, and additionally to a rotating file when
    ``LOG_FILE`` is set. Every line carries the request id of the
    request being handled, which makes concurrent requests traceable.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if any(
        getattr(handler, _HANDLER_MARKER, False)
        for handler in root.handlers
    ):
        return

    formatter = _build_formatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RequestIdFilter())
    setattr(stream_handler, _HANDLER_MARKER, True)
    root.addHandler(stream_handler)

    log_file = os.getenv("LOG_FILE")

    if log_file:
        log_dir = os.path.dirname(log_file)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())
        setattr(file_handler, _HANDLER_MARKER, True)
        root.addHandler(file_handler)
