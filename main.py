import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from app.core.logging_config import request_id_var, setup_logging

setup_logging()


import logging
import re
import time
import uuid
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from openai import OpenAI
import datetime
import chromadb
from app.tools import tool_registry
from app.agent.loop import AgentLoop
from app.agent.request_budget import DEFAULT_MAX_CONTEXT_CHARS
from app.core.database import (
    build_database_url,
    create_engine_safely,
    describe_database_failure,
)
from app.core.rate_limit import RateLimiter
# Re-exported for backwards compatibility (test_main.py imports it
# from this module); the implementation lives in app.core.text.
from app.core.text import is_greeting
from app.planner.planner import Planner
from app.planner.state import TaskState
from app.permissions.checker import PermissionChecker
from app.skills import skill_registry
from app.skills.knowledge import KnowledgeSearchSkill
from app.skills.data_analysis import DataAnalysisSkill
from app.rag.chunker import TextChunker
from app.rag.ingest import ingest_document
from app.memory.manager import MemoryManager
from app.mcp.client import MCPClient
from app.mcp.tool_registry import MCPToolRegistry
from app.core.output_guard import scan_and_log_reply
import bcrypt
import jwt
import redis
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError
from prompts import CHAT_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES = 2
MCP_TOOL_CALL_TIMEOUT_SECONDS = 30.0
APP_MAX_CONTEXT_CHARS = DEFAULT_MAX_CONTEXT_CHARS


def _read_bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable without truthy-string guessing."""
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be a boolean value (true/false, 1/0, yes/no, on/off)"
    )


MCP_AUTO_REFRESH = _read_bool_env("MCP_AUTO_REFRESH", default=False)

BCRYPT_MAX_PASSWORD_BYTES = 72
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300.0
REGISTER_MAX_ATTEMPTS = 5
REGISTER_WINDOW_SECONDS = 3600.0

# /chat drives an LLM call (cost + latency) and /upload_doc writes to
# the database and the vector store, so both are throttled per user on
# top of authentication. Budgets are deliberately generous: they only
# stop runaway loops and scripted abuse, not normal interactive use.
CHAT_MAX_ATTEMPTS = 30
CHAT_WINDOW_SECONDS = 60.0
UPLOAD_MAX_ATTEMPTS = 20
UPLOAD_WINDOW_SECONDS = 60.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

    mcp_client.close()


app = FastAPI(lifespan=lifespan)
security = HTTPBearer()


_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]

if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


REQUEST_ID_MAX_LENGTH = 64
REQUEST_ID_PATTERN = re.compile(
    f"[A-Za-z0-9._-]{{1,{REQUEST_ID_MAX_LENGTH}}}"
)


def request_id_for(request: Request) -> str:
    """Trace id for this request, validated before it is used anywhere.

    A client may suggest an id (useful when a gateway or the client
    already logged one), but this value is written into every log line
    of the request and echoed back in the response header, so it is
    only accepted when it matches a strict whitelist: ASCII letters,
    digits, dot, underscore and hyphen, 1 to 64 characters. Anything
    else — control characters, whitespace, non-ASCII text, or a string
    of unbounded length — is replaced with a server-generated id.

    The returned value is the single id used for both the log records
    (through ``request_id_var``) and the ``X-Request-ID`` response
    header, so what a caller sees back is exactly what was logged.
    """
    candidate = request.headers.get("X-Request-ID")

    if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate

    return uuid.uuid4().hex[:12]


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Give every request a traceable id and log its outcome."""
    request_id = request_id_for(request)

    token = request_id_var.set(request_id)
    started_at = time.perf_counter()
    response = None

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            "http_request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            getattr(response, "status_code", "error"),
            (time.perf_counter() - started_at) * 1000,
        )
        request_id_var.reset(token)


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout=LLM_TIMEOUT_SECONDS,
    max_retries=LLM_MAX_RETRIES,
)


permission_checker = PermissionChecker()


CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    "./chroma_data",
)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIRECTORY,
)
collection = chroma_client.get_or_create_collection(name="my_docs")

skill_registry.register(KnowledgeSearchSkill(collection, tool_registry))

skill_registry.register(
    DataAnalysisSkill(tool_registry)
)

mcp_client = MCPClient(
    call_timeout=MCP_TOOL_CALL_TIMEOUT_SECONDS,
)
mcp_tool_registry = MCPToolRegistry(mcp_client)

if MCP_AUTO_REFRESH:
    try:
        mcp_tool_registry.refresh()
    except Exception:
        logger.exception(
            "mcp_refresh_failed: continuing without MCP tools"
        )

planner = Planner(
    client=client,
    tool_registry=tool_registry,
    skill_registry=skill_registry,
    mcp_tool_registry=mcp_tool_registry,
    permission_checker=permission_checker,
    max_context_chars=APP_MAX_CONTEXT_CHARS,
)

agent_loop = AgentLoop(
    client=client,
    tool_registry=tool_registry,
    skill_registry=skill_registry,
    permission_checker=permission_checker,
    planner=planner,
    task_state_class=TaskState,
    mcp_tool_registry=mcp_tool_registry,
    max_context_chars=APP_MAX_CONTEXT_CHARS,
)

def build_redis_client():
    """Build the Redis client from environment configuration.

    ``REDIS_PASSWORD`` is optional: when it is unset or blank, no AUTH
    is attempted, so password-less deployments keep working. The value
    is never hardcoded and is only read from the environment.
    """
    raw_password = os.getenv("REDIS_PASSWORD")
    password = raw_password.strip() if raw_password else ""

    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=password or None,
        decode_responses=True,
        protocol=2,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


redis_client = build_redis_client()


SECRET_KEY = os.environ["JWT_SECRET_KEY"]

# The signing key comes from the environment (.env is loaded at
# import). Only its length is ever logged — never the value — so an
# operator can spot a key that is too short for HS256 without it
# leaking into logs or error messages.
if len(SECRET_KEY.encode("utf-8")) < 32:
    logger.warning(
        "jwt_secret_key_short length=%d_bytes recommended>=32",
        len(SECRET_KEY.encode("utf-8")),
    )



MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "agent_db")

# Built and created through app.core.database so that a missing DB
# driver or an unparseable DSN fails with a classified, secret-free
# error instead of an import-time traceback. The password is
# percent-encoded inside build_database_url, so special characters
# cannot break the URL parse. The URL itself is deliberately not kept
# as a module attribute: nothing should be able to print it by
# accident (e.g. a careless repr in a log or an error handler).
engine = create_engine_safely(
    build_database_url(
        MYSQL_USER,
        os.environ["MYSQL_PASSWORD"],
        MYSQL_HOST,
        MYSQL_PORT,
        database=MYSQL_DATABASE,
    ),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    password_hash = Column(String(255))
    role = Column(String(20), default="employee")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    role = Column(String(20))
    content = Column(Text)
    user_id = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True)
    summary_text = Column(Text)
    covers_up_to_message_id = Column(Integer)
    pending_message_id = Column(Integer)
    pending_content_hash = Column(String(64))
    pending_offset = Column(Integer, nullable=False, default=0)
    pending_summary_text = Column(Text)
    failure_count = Column(Integer, nullable=False, default=0)
    next_retry_at = Column(DateTime)
    failure_status = Column(String(20))
    failure_code = Column(String(64))
    version = Column(Integer, nullable=False, default=0)


COMPRESSION_THRESHOLD = 20


if os.getenv("AUTO_CREATE_TABLES", "true").lower() in ("1", "true", "yes"):
    # Creates any missing tables (never alters existing ones) so a
    # fresh database works without manual DDL. Disable with
    # AUTO_CREATE_TABLES=false in environments with migrations.
    try:
        Base.metadata.create_all(engine)
        logger.info("database_tables_ensured")
    except Exception as exc:
        # Deliberately not logger.exception: a full traceback would
        # print the driver's message verbatim, and driver messages are
        # not covered by any redaction contract. The category plus a
        # redacted, truncated detail is enough to tell "server down"
        # from "bad credentials" without risking a DSN fragment.
        logger.error(
            "database_table_creation_failed %s",
            describe_database_failure(exc),
        )


memory_manager = MemoryManager(
    redis_client,
    client,
    Message,
    ConversationSummary,
    max_context_chars=APP_MAX_CONTEXT_CHARS,
)



def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def split_text(text, chunk_size=300, overlap=80):
    return [
        chunk.text
        for chunk in TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        ).split(text)
    ]


def add_document(text, permission, title=None, uploaded_by=None):
    return ingest_document(
        collection,
        text,
        permission=permission,
        title=title,
        uploaded_by=uploaded_by,
    )


def build_chat_response(reply, include_plan=False, task_state=None):
    payload = {"reply": reply}

    if include_plan:
        payload["plan"] = (
            task_state.as_dict() if task_state is not None else None
        )

    return payload


def hash_password(password):
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed.decode("utf-8")


def verify_password(password, hashed):
    try:
        password_bytes = password.encode("utf-8")
        hashed_bytes = hashed.encode("utf-8")

        return bcrypt.checkpw(
            password_bytes,
            hashed_bytes
        )
    except (ValueError, TypeError):
        # A corrupt or empty stored hash must not crash login;
        # it simply counts as a failed verification.
        logger.warning("password_verify_failed malformed_stored_hash")

        return False


def create_token(user_id, role):
    payload = {
        "user_id": user_id,
        "role": role,
        # Timezone-aware UTC: datetime.utcnow() is deprecated, and a
        # naive datetime would be interpreted in local time by
        # consumers that assume awareness. Semantics are unchanged
        # (both denote the same UTC instant).
        "exp": (
            datetime.datetime.now(datetime.UTC)
            + datetime.timedelta(hours=24)
        ),
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm="HS256"
    )

    return token


def decode_token(token):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"]
        )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired"
        )

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    payload = decode_token(token)

    return {
        "user_id": payload["user_id"],
        "role": payload["role"],
    }


def get_current_db_user(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Load the authoritative user row for this request.

    The role is read from the database instead of the JWT so that
    role changes and account deletions take effect immediately,
    even while already-issued tokens are still valid.
    """
    user = db.query(User).filter(
        User.id == current_user["user_id"]
    ).first()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    return user


class ChatMessage(BaseModel):
    message: str = Field(max_length=8000)
    include_plan: bool = False


class DocumentUpload(BaseModel):
    text: str = Field(max_length=100_000)
    permission: Literal["public", "employee", "finance", "admin"]
    title: str | None = Field(default=None, max_length=200)


class UserRegister(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^\S+$",
    )
    password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_must_fit_bcrypt(cls, value: str) -> str:
        # bcrypt only uses the first 72 bytes and raises for longer
        # input, so reject oversized passwords with a clear 422
        # instead of a 500 later.
        if len(value.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
            raise ValueError(
                "password must be at most "
                f"{BCRYPT_MAX_PASSWORD_BYTES} bytes"
            )

        return value


class UserLogin(BaseModel):
    username: str
    password: str


login_limiter = RateLimiter(
    max_attempts=LOGIN_MAX_ATTEMPTS,
    window_seconds=LOGIN_WINDOW_SECONDS,
)

register_limiter = RateLimiter(
    max_attempts=REGISTER_MAX_ATTEMPTS,
    window_seconds=REGISTER_WINDOW_SECONDS,
)

chat_limiter = RateLimiter(
    max_attempts=CHAT_MAX_ATTEMPTS,
    window_seconds=CHAT_WINDOW_SECONDS,
)

upload_limiter = RateLimiter(
    max_attempts=UPLOAD_MAX_ATTEMPTS,
    window_seconds=UPLOAD_WINDOW_SECONDS,
)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def login_rate_key(ip: str) -> str:
    """Budget key for login attempts, scoped to the client IP.

    The username is deliberately NOT part of the key: keying on
    ``ip + username`` let an attacker reset the budget simply by
    switching to another account, so one IP could probe unlimited
    usernames. Per-account budgets (from many IPs) are intentionally
    not enforced here, because locking an account would let anyone
    deny its owner access.
    """
    return f"login:{ip}"


def register_rate_key(ip: str) -> str:
    return f"register:{ip}"


def chat_rate_key(user_id) -> str:
    return f"chat:{user_id}"


def upload_rate_key(user_id) -> str:
    return f"upload:{user_id}"


def _too_many_requests(detail: str, retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(retry_after)},
    )


@app.post("/register")
def register(
    data: UserRegister,
    request: Request,
    db=Depends(get_db)
):
    ip = client_ip(request)

    allowed, retry_after = register_limiter.hit(
        register_rate_key(ip)
    )

    if not allowed:
        raise _too_many_requests(
            "Too many registration attempts, try again later",
            retry_after,
        )

    new_user = User(
        username=data.username,
        password_hash=hash_password(data.password)
    )

    db.add(new_user)

    try:
        db.commit()

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    return {
        "message": "User registered successfully"
    }


@app.post("/login")
def login(
    data: UserLogin,
    request: Request,
    db=Depends(get_db)
):
    ip = client_ip(request)
    rate_key = login_rate_key(ip)

    allowed, retry_after = login_limiter.hit(rate_key)

    if not allowed:
        raise _too_many_requests(
            "Too many login attempts, try again later",
            retry_after,
        )

    user = db.query(User).filter(
        User.username == data.username
    ).first()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    if not verify_password(
        data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    login_limiter.reset(rate_key)

    token = create_token(user.id, user.role)

    return {
        "access_token": token
    }


@app.post("/upload_doc")
def upload_doc(
    data: DocumentUpload,
    current_user: User = Depends(get_current_db_user)
):
    role = current_user.role
    user_id = current_user.id

    # Throttle before any database or vector-store work happens.
    allowed, retry_after = upload_limiter.hit(upload_rate_key(user_id))

    if not allowed:
        raise _too_many_requests(
            "Too many document uploads, try again later",
            retry_after,
        )

    if not permission_checker.can_upload_document(
        role,
        data.permission,
    ):
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to upload this document level"
        )

    try:
        add_document(
            data.text,
            data.permission,
            title=data.title,
            uploaded_by=user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return {
        "message": "Document stored"
    }


@app.post("/chat")
def chat(
    data: ChatMessage,
    current_user: dict = Depends(get_current_user),
    db_user: User = Depends(get_current_db_user),
    db=Depends(get_db)
):
    user_id = db_user.id
    role = db_user.role

    # Reject runaway clients before the LLM call and before any row
    # is written, so a throttled request leaves no partial history.
    allowed, retry_after = chat_limiter.hit(chat_rate_key(user_id))

    if not allowed:
        raise _too_many_requests(
            "Too many chat requests, please slow down",
            retry_after,
        )

    user_message = Message(
        role="user",
        content=data.message,
        user_id=str(user_id)
    )

    db.add(user_message)
    db.commit()

    chat_history = memory_manager.get_chat_history(
        user_id,
        db
    )

    context_text = (
        "Reference material will be retrieved by the "
        "knowledge_search skill when needed."
    )
    chat_history.insert(
        0,
        {
            "role": "system",
            "content": CHAT_SYSTEM_PROMPT.format(
                context_text=context_text
            )
        }
    )

    # Agent Loop
    try:
        reply, task_state = agent_loop.run(
            chat_history,
            role=role,
        )
    except Exception:
        logger.exception(
            "chat_agent_failed user_id=%s",
            user_id,
        )

        fallback_message = Message(
            role="assistant",
            content=(
                "（系统）当前无法处理该请求，请稍后重试。"
            ),
            user_id=str(user_id)
        )

        db.add(fallback_message)
        db.commit()

        raise HTTPException(
            status_code=503,
            detail="Agent processing failed, please try again later"
        )

    # Best-effort output monitoring only: records a log event when the
    # reply looks like it leaked credentials or echoed injection
    # framing. Never blocks or rewrites the answer.
    scan_and_log_reply(reply)

    assistant_message = Message(
        role="assistant",
        content=reply,
        user_id=str(user_id)
    )

    db.add(assistant_message)
    db.commit()

    try:
        memory_manager.maybe_compress_history(
            user_id,
            db
        )
    except Exception:
        db.rollback()
        logger.exception(
            "history_compression_failed user_id=%s",
            user_id,
        )

    return build_chat_response(
        reply,
        include_plan=data.include_plan,
        task_state=task_state,
    )




























