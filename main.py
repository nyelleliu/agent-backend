import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from openai import OpenAI
import json
import datetime
import chromadb
import bcrypt
import jwt
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError

app = FastAPI()
security = HTTPBearer()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="my_docs")

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

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    role = Column(String(20))
    content = Column(Text)
    user_id = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_time():
    now = datetime.datetime.now()
    weekday_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return f"Now is {now.strftime('%Y-%m-%d %H:%M:%S')}, {weekday_map[now.weekday()]}"

def calculate(expression):
    try:
        a = eval(expression)
        return a
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"

def split_text(text, chunk_size=300):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        start += chunk_size
    return chunks

def add_document(text):
    chunks = split_text(text)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

def hash_password(password):
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")

def verify_password(password, hashed):
    password_bytes = password.encode("utf-8")
    hashed_bytes = hashed.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)

def create_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    return payload["user_id"]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date, time and day of week",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a math expression, e.g. addition, subtraction, multiplication, division",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to calculate, e.g. 23 * 47"
                    }
                }
            }
        }
    }
]

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
def register(data: UserRegister, db = Depends(get_db)):
    new_user = User(username=data.username, password_hash=hash_password(data.password))
    db.add(new_user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")

    return {"message": "User registered successfully"}

@app.post("/login")
def login(data: UserLogin, db = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(user.id)
    return {"access_token": token}

@app.post("/upload_doc")
def upload_doc(data: DocumentUpload, user_id: int = Depends(get_current_user)):
    add_document(data.text)
    return {"message": "Document stored"}

@app.post("/chat")
def chat(data: ChatMessage, user_id: int = Depends(get_current_user), db = Depends(get_db)):
    user_message = Message(role="user", content=data.message, user_id=str(user_id))
    db.add(user_message)
    db.commit()

    history_rows = db.query(Message).filter(Message.user_id == str(user_id)).order_by(Message.id).all()
    chat_history = [{"role": m.role, "content": m.content} for m in history_rows]

    results = collection.query(
        query_texts=[data.message],
        n_results=3,
        include=["documents", "distances"]
    )
    distances = results["distances"][0]
    documents = results["documents"][0]

    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Query: {data.message}\n")
        f.write(f"Distances: {distances}\n")
        f.write(f"Documents: {documents}\n\n")

    filtered_chunks = []
    for i in range(len(documents)):
        if distances[i] < 1.5:
            filtered_chunks.append(documents[i])

    context_text = "\n".join(filtered_chunks) if filtered_chunks else "No relevant reference material found."

    chat_history.insert(0, {
        "role": "system",
        "content": f"Reference material that may or may not be relevant:\n{context_text}\nIf it isn't relevant to the question, ignore it."
    })

    reply = "Sorry, I couldn't complete this after several tool calls."

    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=chat_history,
            tools=tools,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            reply = msg.content
            break

        chat_history.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            if tc.function.name == "get_current_time":
                result = get_current_time()
            elif tc.function.name == "calculate":
                args = json.loads(tc.function.arguments)
                expression = args["expression"]
                result = calculate(expression)
            else:
                result = "unknown tool"

            chat_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    assistant_message = Message(role="assistant", content=reply, user_id=str(user_id))
    db.add(assistant_message)
    db.commit()

    return {"reply": reply}


