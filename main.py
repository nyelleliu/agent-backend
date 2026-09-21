import os
import json
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from openai import OpenAI
import datetime
import chromadb
from app.tools import tool_registry
from app.agent.loop import AgentLoop
from app.planner.planner import Planner
from app.planner.state import TaskState
from app.permissions.checker import PermissionChecker
from app.skills import skill_registry
from app.skills.knowledge import KnowledgeSearchSkill
from app.skills.data_analysis import DataAnalysisSkill
from app.memory.manager import MemoryManager
import bcrypt
import jwt
import redis
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError
from prompts import CHAT_SYSTEM_PROMPT


app = FastAPI()
security = HTTPBearer()


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)


permission_checker = PermissionChecker()


chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="my_docs")

skill_registry.register(KnowledgeSearchSkill(collection, tool_registry))

skill_registry.register(
    DataAnalysisSkill(tool_registry)
)

planner = Planner(
    client=client,
    tool_registry=tool_registry,
    skill_registry=skill_registry,
)

agent_loop = AgentLoop(
    client=client,
    tool_registry=tool_registry,
    skill_registry=skill_registry,
    permission_checker=permission_checker,
    planner=planner,
    task_state_class=TaskState,
)

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
    protocol=2
)


SECRET_KEY = os.environ["JWT_SECRET_KEY"]


DATABASE_URL = f"mysql+pymysql://root:{os.environ['MYSQL_PASSWORD']}@localhost/agent_db"
engine = create_engine(DATABASE_URL)
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


COMPRESSION_THRESHOLD = 20


memory_manager = MemoryManager(
    redis_client,
    client,
    Message,
    ConversationSummary,
)



def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def is_greeting(message):
    greetings = ["你好", "hi", "hello", "嗨", "早上好", "晚上好", "在吗"]
    return any(g in message.lower() for g in greetings)


def split_text(text, chunk_size=300):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size

    return chunks


def add_document(text):
    chunks = split_text(text)
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids
    )


def hash_password(password):
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed.decode("utf-8")


def verify_password(password, hashed):
    password_bytes = password.encode("utf-8")
    hashed_bytes = hashed.encode("utf-8")

    return bcrypt.checkpw(
        password_bytes,
        hashed_bytes
    )


def create_token(user_id, role):
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
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

    return payload["user_id"]


class ChatMessage(BaseModel):
    message: str


class DocumentUpload(BaseModel):
    text: str


class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


@app.post("/register")
def register(
    data: UserRegister,
    db=Depends(get_db)
):
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
    db=Depends(get_db)
):
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

    token = create_token(user.id, user.role)

    return {
        "access_token": token
    }


@app.post("/upload_doc")
def upload_doc(
    data: DocumentUpload,
    user_id: int = Depends(get_current_user)
):
    add_document(data.text)

    return {
        "message": "Document stored"
    }


@app.post("/chat")
def chat(
    data: ChatMessage,
    user_id: int = Depends(get_current_user),
    db=Depends(get_db)
):
    user = db.query(User).filter(
        User.id == user_id
    ).first()

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found"
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
    reply = agent_loop.run(
        chat_history,
        role=user.role,
    )

    assistant_message = Message(
        role="assistant",
        content=reply,
        user_id=str(user_id)
    )

    db.add(assistant_message)
    db.commit()

    memory_manager.maybe_compress_history(
        user_id,
        db
    )

    return {
        "reply": reply
    }


























