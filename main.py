import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import pymysql
import json
import datetime
import chromadb

app = FastAPI()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_docs")

def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password=os.environ["MYSQL_PASSWORD"],
        database="agent_db",
        charset="utf8mb4"
    )

def get_current_time():
    now = datetime.datetime.now()
    weekday_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return f"Now is {now.strftime('%Y-%m-%d %H:%M:%S')}, {weekday_map[now.weekday()]}"

def calculate(expression):
    a = eval(expression)
    return a

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

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date, time and day of week",
            "parameters": {
                "type": "object",
                "properties": {}
            }
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
    user_id: str

class DocumentUpload(BaseModel):
    text: str

@app.post("/upload_doc")
def upload_doc(data: DocumentUpload):
    add_document(data.text)
    return {"message": "Document stored"}

@app.post("/chat")
def chat(data: ChatMessage):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO messages (role, content, user_id) VALUES (%s, %s, %s)",
        ("user", data.message, data.user_id)
    )
    conn.commit()

    cursor.execute(
        "SELECT role, content FROM messages WHERE user_id = %s ORDER BY id",
        (data.user_id,)
    )
    rows = cursor.fetchall()
    chat_history = [{"role": r[0], "content": r[1]} for r in rows]

    results = collection.query(query_texts=[data.message], n_results=3)
    retrieved_chunks = results["documents"][0]
    context_text = "\n".join(retrieved_chunks)

    chat_history.insert(0, {
        "role": "system",
        "content": f"Reference material that may or may not be relevant:\n{context_text}\nIf it isn't relevant to the question, ignore it."
    })

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=chat_history,
        tools=tools,
    )
    msg = response.choices[0].message

    if msg.tool_calls:
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

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=chat_history,
        )
        reply = response.choices[0].message.content
    else:
        reply = msg.content

    cursor.execute(
        "INSERT INTO messages (role, content, user_id) VALUES (%s, %s, %s)",
        ("assistant", reply, data.user_id)
    )
    conn.commit()

    cursor.close()
    conn.close()

    return {"reply": reply}
