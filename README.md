# Agent Backend

一个基于 FastAPI + DeepSeek + ChromaDB + Redis + MySQL 构建的企业内部智能查询 Agent。

项目目标不是简单实现一个 Chatbot，而是逐步构建一个具备 **Tool Calling、Agent Loop、RAG、Memory、Skill、用户认证** 等能力的 Agent 后端。

## 当前版本

**v0.2 — Agent Architecture Refactor**

当前已经完成：

- Tool Registry
- Agent Loop
- Skill Registry
- Knowledge Search Skill
- RAG 文档检索
- MySQL 持久化对话
- Redis 对话摘要缓存
- JWT 用户认证
- 自动化测试

---

## 项目架构

```text
Agent Backend
│
├── main.py
│   └── FastAPI API
│
├── app/
│   ├── agent/
│   │   └── loop.py
│   │       └── Agent Loop
│   │
│   ├── tools/
│   │   ├── registry.py
│   │   ├── calculator.py
│   │   └── datetime.py
│   │       └── Tool Registry
│   │
│   └── skills/
│       ├── base.py
│       ├── registry.py
│       └── knowledge.py
│           └── Knowledge Search Skill
│
├── tests/
│   ├── test_main.py
│   ├── test_tools.py
│   └── test_skills.py
│
├── chroma_data/
│   └── Persistent Vector Database
│
└── MySQL
    ├── Users
    ├── Messages
    └── Conversation Summaries
```

---

## Agent 工作流程

```text
用户请求
   ↓
FastAPI
   ↓
JWT 身份认证
   ↓
加载用户历史消息
   ↓
RAG / Skill
   ↓
Agent Loop
   ↓
LLM 判断是否需要调用 Tool / Skill
   │
   ├── 不需要
   │      ↓
   │   直接生成回答
   │
   └── 需要
          ↓
      Tool / Skill Registry
          ↓
      执行工具
          ↓
      返回执行结果
          ↓
      再次调用 LLM
          ↓
      最终回答
   ↓
保存对话
   ↓
返回用户
```

Agent 的核心思想是：

> **LLM 负责决策，Backend 负责执行。**

LLM 不直接执行 Python、数据库查询或其他后端操作，而是通过 Tool Calling 请求后端执行具体能力。

---

## Tool Registry

项目已经将原来的工具函数重构为统一的 Tool Registry。

目前包含：

### `calculate`

执行数学表达式，例如：

```text
23 * 47
```

底层使用 AST 解析，而不是直接使用 `eval()`，避免执行任意 Python 代码。

### `get_current_time`

返回当前日期、时间和星期。

Tool Registry 负责：

```text
Tool
 ↓
注册
 ↓
Schema
 ↓
LLM Tool Calling
 ↓
参数解析与校验
 ↓
执行
 ↓
返回结果
```

这样以后增加工具时，不需要继续在 Agent 中堆积大量 `if / elif`。

---

## Agent Loop

Agent Loop 位于：

```text
app/agent/loop.py
```

核心逻辑：

```text
LLM
 ↓
是否产生 tool_call？
 │
 ├── 否 → 返回最终答案
 │
 └── 是
      ↓
   执行 Tool / Skill
      ↓
   将结果加入 messages
      ↓
   再次请求 LLM
      ↓
   最终答案
```

当前最大执行步数为 `5`，用于避免 Agent 无限循环。

---

## Skill Architecture

除了基础 Tool 之外，项目还引入了 Skill。

目前实现：

```text
SkillRegistry
      │
      └── KnowledgeSearchSkill
```

### Knowledge Search Skill

用于查询企业知识库。

调用流程：

```text
用户问题
   ↓
KnowledgeSearchSkill
   ↓
Chroma 查询 Top 3
   ↓
距离过滤
   ↓
返回相关文档片段
   ↓
LLM 生成最终回答
```

当前代码使用的距离阈值：

```text
distance < 1.5
```

Tool 和 Skill 在架构上分开管理，但当前都会通过 OpenAI-compatible Tool Calling 机制暴露给 LLM。

---

## RAG

项目使用 ChromaDB 作为持久化向量数据库。

文档处理流程：

```text
原始文档
   ↓
文本切分
   ↓
Chunk
   ↓
Chroma Embedding
   ↓
Vector Database
```

查询流程：

```text
用户问题
   ↓
Chroma Query
   ↓
Top 3 chunks
   ↓
距离过滤
   ↓
相关文档
   ↓
Agent / LLM
```

当前项目使用简单的固定长度文本切分：

```text
chunk_size = 300
```

Chroma 数据持久化在：

```text
./chroma_data
```

该目录不提交到 Git。

---

## Conversation Memory

项目目前使用：

```text
MySQL + Redis
```

保存和管理对话记忆。

### MySQL

保存：

- 用户
- 对话消息
- Conversation Summary

### Redis

用于缓存 Conversation Summary，减少频繁访问数据库。

当历史消息达到一定数量后：

```text
大量历史消息
      ↓
LLM Summary
      ↓
ConversationSummary
      ↓
Redis Cache
      ↓
后续对话继续使用
```

当前压缩阈值：

```text
COMPRESSION_THRESHOLD = 20
```

---

## 用户认证

项目已经加入 JWT 用户认证。

核心流程：

```text
Register
   ↓
密码 bcrypt Hash
   ↓
MySQL

Login
   ↓
验证密码
   ↓
生成 JWT
   ↓
访问需要认证的 API
```

JWT 中包含用户身份信息，并使用过期时间控制 Token 有效期。

因此不同用户的对话数据可以进行隔离。

---

## API

当前主要 API：

```text
POST /register
POST /login
POST /upload_doc
POST /chat
```

完整接口可以通过 FastAPI Swagger 查看：

```text
http://127.0.0.1:8000/docs
```

---

## Tech Stack

| 技术 | 用途 |
|---|---|
| FastAPI | Web API |
| DeepSeek API | LLM |
| OpenAI SDK | 调用 OpenAI-compatible API |
| ChromaDB | 向量数据库 / RAG |
| MySQL | 用户、消息、摘要持久化 |
| Redis | Memory Summary 缓存 |
| SQLAlchemy | ORM |
| JWT | 用户认证 |
| bcrypt | 密码 Hash |
| Pydantic | 数据校验 |
| pytest | 自动化测试 |

---

## 项目测试

当前项目已经加入：

- Tool Registry 测试
- Calculator 安全测试
- Skill Registry 测试
- FastAPI API 测试
- RAG 文本切分测试

当前测试结果：

```text
19 passed
```

---

## 项目演进

### v0.1 — Basic Agent

最初版本主要实现：

```text
FastAPI
  ↓
LLM
  ↓
Tool Calling
  ↓
Calculator / Current Time
```

同时加入了基础 RAG 和 MySQL Conversation Memory。

### v0.2 — Agent Architecture Refactor

当前版本重点进行 Agent 架构重构：

```text
Tool
  ↓
Tool Registry

Skill
  ↓
Skill Registry

LLM
  ↓
Agent Loop
  ↓
Tool / Skill
```

使项目从一个简单的 AI API，逐渐演变成模块化 Agent Backend。

---

## Roadmap

后续计划：

```text
v0.2
├── Tool Registry ✓
├── Agent Loop ✓
├── Skill Registry ✓
└── Knowledge Search Skill ✓

        ↓

v0.3
├── RAG Architecture Improvement
├── Permission Control
├── Better Memory
└── Evaluation

        ↓

v0.4
├── Planning
├── MCP
└── Multi-Agent

        ↓

v1.0
└── Enterprise AI Agent Platform
```

最终目标：

> 构建一个能够查询企业知识、调用业务工具、分析数据，并根据用户权限完成多步骤任务的企业级 AI Agent。

---

## Learning Goals

这个项目同时作为 Agent 学习项目。

通过实际代码理解：

```text
LLM
 ↓
Tool Calling
 ↓
Agent Loop
 ↓
Tool Registry
 ↓
Skill
 ↓
RAG
 ↓
Memory
 ↓
Planning
 ↓
MCP
 ↓
Multi-Agent
```

最终能够从工程实现角度回答 Agent 实习面试中的核心问题。

---

## Setup

### Requirements

```text
Python 3.10+
MySQL
Redis
DeepSeek API Key
```

### Install

```bash
pip install fastapi uvicorn pymysql openai chromadb redis sqlalchemy bcrypt python-jose pytest
```

### Environment Variables

```text
DEEPSEEK_API_KEY
MYSQL_PASSWORD
JWT_SECRET
```

### Run

```bash
uvicorn main:app --reload
```

API：

```text
http://127.0.0.1:8000
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

---

## Project Status

当前项目已经完成：

```text
✓ FastAPI Backend
✓ DeepSeek LLM
✓ Tool Calling
✓ Tool Registry
✓ Agent Loop
✓ Skill Registry
✓ Knowledge Search Skill
✓ RAG
✓ MySQL Memory
✓ Redis Summary Cache
✓ JWT Authentication
✓ Automated Tests
```

项目正在从：

```text
简单 AI Chat Backend
```

逐步演进为：

```text
企业内部智能查询 Agent
```
