# Agent Backend

A FastAPI backend service for an AI agent with tool calling, RAG-based document Q&A, persistent conversation memory, and multi-user support.

一个基于 FastAPI 的 AI Agent 后端服务，支持工具调用、RAG文档问答、持久化对话记忆和多用户隔离。

## Overview / 项目简介

This project implements a conversational AI backend that goes beyond a simple chat wrapper. It follows the core agent architecture pattern — the model can request tool calls (e.g. calculation, current time), the server executes them, and the results are fed back to the model before producing a final response. It also supports RAG (Retrieval-Augmented Generation): users can upload documents, which are chunked and embedded into a vector store (Chroma), and relevant chunks are retrieved and injected into the conversation whenever a question is asked — letting the agent answer questions about private/custom knowledge it was never trained on. Conversation history is persisted in MySQL rather than kept in memory, so context survives server restarts. Each user's conversation is isolated by `user_id`.

这个项目实现的不只是一个简单的对话接口，而是完整的 agent 架构模式：模型可以请求调用工具（比如计算、查询当前时间），服务器负责真正执行，再把结果返回给模型生成最终回复。项目同时支持 RAG（检索增强生成）：用户可以上传文档，文档会被切片并向量化存入向量数据库（Chroma），每次提问时会检索相关片段并注入对话上下文，使 agent 能够回答训练数据中不存在的私有/自定义知识问题。对话历史存储在 MySQL 而不是内存里，因此服务重启后上下文依然保留。每个用户的对话通过 `user_id` 相互隔离。

## Features / 功能特点

- **RAG document Q&A** — upload arbitrary text documents; they're chunked, embedded, and stored in a Chroma vector store. On each chat request, the most relevant chunks are retrieved and injected as context, letting the agent answer questions about content it was never trained on.
  **RAG 文档问答** —— 上传任意文本文档，系统自动切片、向量化并存入 Chroma 向量数据库。每次对话时检索最相关的片段并注入上下文，使 agent 能回答训练数据之外的私有知识问题。

- **Multi-turn conversation with persistent memory** — chat history is stored in MySQL and reloaded on every request, so context is never lost even after a server restart.
  **多轮对话与持久化记忆** —— 对话历史存储在 MySQL 中，每次请求都会重新加载，即使服务重启，上下文也不会丢失。

- **Tool calling (agent loop)** — the LLM can call registered tools (a calculator and a date/time lookup) when it needs information it can't reliably produce itself. This avoids the common failure mode of LLMs miscalculating large numbers.
  **工具调用（agent 循环）** —— 大模型在需要自己无法可靠得出的信息时，可以调用已注册的工具（计算器、日期时间查询），避免了大模型心算大数字容易出错的常见问题。

- **Multi-user isolation** — conversations are scoped by `user_id`, so different users' histories never mix.
  **多用户隔离** —— 对话按 `user_id` 区分，不同用户的历史记录互不干扰。

- **Secrets managed via environment variables** — API keys and database credentials are never hardcoded; they're read from environment variables at runtime.
  **敏感信息通过环境变量管理** —— API key 和数据库密码从不写死在代码里，运行时从环境变量读取。

## Tech Stack / 技术栈

- **FastAPI** — web framework and REST API / Web 框架与接口层
- **MySQL**（通过 `pymysql`） — persistent storage for conversation history / 持久化存储对话历史
- **Chroma** — vector database for RAG document storage and retrieval / RAG 文档存储与检索的向量数据库
- **DeepSeek API**（OpenAI 兼容接口） — the underlying LLM / 底层大模型
- **Pydantic** — request validation / 请求数据校验

## Architecture / 架构流程

### Chat flow / 对话流程

```
User request 用户请求
    ↓
Save user message to MySQL 存入用户消息
    ↓
Load full conversation history for this user_id 读取该用户完整历史
    ↓
Retrieve top-3 relevant chunks from Chroma based on the question
根据问题从 Chroma 中检索最相关的3个片段
    ↓
Inject retrieved chunks as a system message 将检索结果作为system消息注入
    ↓
Send history + tool definitions + retrieved context to the LLM
发送历史+工具清单+检索到的上下文给大模型
    ↓
   Does the model request a tool call? 模型是否请求工具调用？
   ├─ No 否  → return the model's text reply 直接返回文字回复
   └─ Yes 是 → execute the requested tool(s) locally 本地执行工具
              ↓
           feed tool results back into the conversation 把结果塞回对话
              ↓
           call the LLM again for a final reply 再次调用模型拿最终回复
    ↓
Save assistant reply to MySQL 存入assistant回复
    ↓
Return reply to user 返回给用户
```

This mirrors the core loop used by coding agents like Claude Code / CoreCoder: the LLM never executes anything itself — it only requests actions, and the backend is responsible for executing them and feeding results back.

这套循环跟 Claude Code / CoreCoder 这类编程 agent 的核心逻辑是一致的：大模型自己从不真正执行任何操作，它只是"请求"动作，真正的执行和结果反馈由后端负责。

### Document ingestion flow / 文档入库流程

```
Raw document text 原始文档
    ↓
Split into fixed-size chunks (300 chars, sliding window) 切分为固定大小的片段
    ↓
Each chunk is embedded into a vector (handled automatically by Chroma)
每个片段被转换为向量（Chroma自动完成）
    ↓
Stored in the Chroma collection with a unique id
以唯一id存入Chroma集合
```

## API

### `POST /chat`

**Request body / 请求体：**
```json
{
  "message": "What's 8947 * 2368?",
  "user_id": "user_123"
}
```

**Response / 返回：**
```json
{
  "reply": "8947 × 2368 = 21,186,496."
}
```

### `POST /upload_doc`

**Request body / 请求体：**
```json
{
  "text": "Full text of a document to be chunked and stored for retrieval."
}
```

**Response / 返回：**
```json
{
  "message": "Document stored"
}
```

## Setup / 环境搭建

### Prerequisites / 前置要求
- Python 3.10+
- MySQL 8.0+
- A DeepSeek API key（或任意 OpenAI 兼容的模型接口）

### Database setup / 数据库准备

```sql
CREATE DATABASE agent_db;
USE agent_db;

CREATE TABLE messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    role VARCHAR(20),
    content TEXT,
    user_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Install dependencies / 安装依赖

```bash
pip install fastapi uvicorn pymysql openai chromadb
```

### Set environment variables / 设置环境变量

```bash
export DEEPSEEK_API_KEY="your-api-key"
export MYSQL_PASSWORD="your-mysql-password"
```

（Windows PowerShell 下用：`$env:DEEPSEEK_API_KEY="..."`）

### Run / 启动服务

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

服务启动后可通过 `http://127.0.0.1:8000` 访问，交互式接口文档在 `http://127.0.0.1:8000/docs`。

## Available Tools / 已实现的工具

| Tool 工具 | Description 说明 |
|---|---|
| `calculate` | Evaluates a math expression（如 `"23 * 47"`）—— 避免大模型自己心算大数字时出错。 |
| `get_current_time` | Returns the current date, time and day of the week —— 大模型自身无法得知的实时信息。 |

## Known Limitations / 已知局限

- **RAG retrieval always returns results, even irrelevant ones** — Chroma returns the top-N nearest chunks regardless of how relevant they actually are. The current mitigation is a prompt instruction telling the model to ignore irrelevant context; a stricter approach would add a similarity-score threshold.
  **RAG 检索总会返回结果，即使不相关** —— Chroma 默认返回最接近的N个片段，不管实际相关程度如何。目前的缓解方式是在提示词中告知模型忽略不相关内容；更严格的做法是加入相似度阈值过滤。
- **No user authentication** — `user_id` is currently supplied directly by the client with no verification.
  **暂无用户认证** —— `user_id` 目前由调用方直接传入，没有验证机制。
- **Chroma runs in-memory** — the vector store is not persisted to disk in the current setup; documents need to be re-uploaded after a restart.
  **Chroma 当前为内存模式** —— 向量数据库未持久化到磁盘，服务重启后需要重新上传文档。
- **Embedding uses Chroma's default local model** — sufficient for demonstration purposes, but a production system would likely use a dedicated embedding API for better retrieval accuracy.
  **Embedding 使用 Chroma 自带的本地默认模型** —— 用于演示已经足够，生产环境通常会使用专门的 embedding API 以获得更好的检索精度。

## Possible Extensions / 可扩展方向

- User authentication（补充登录鉴权）
- Persistent vector storage（让 Chroma 向量数据持久化，而非仅存于内存）
- Similarity threshold filtering for retrieval（检索相似度阈值过滤）
- Additional tools（网页搜索、天气 API 等）
- Streaming responses（流式返回）
- Rate limiting / usage tracking per user（限流与用量统计）
