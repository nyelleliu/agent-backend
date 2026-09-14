# Agent Backend

A FastAPI backend service for an AI agent with tool calling, persistent conversation memory, and multi-user support.

一个基于 FastAPI 的 AI Agent 后端服务，支持工具调用、持久化对话记忆和多用户隔离。

## Overview / 项目简介

This project implements a conversational AI backend that goes beyond a simple chat wrapper. It follows the core agent architecture pattern — the model can request tool calls (e.g. calculation, current time), the server executes them, and the results are fed back to the model before producing a final response. Conversation history is persisted in MySQL rather than kept in memory, so context survives server restarts. Each user's conversation is isolated by `user_id`.

这个项目实现的不只是一个简单的对话接口，而是完整的 agent 架构模式：模型可以请求调用工具（比如计算、查询当前时间），服务器负责真正执行，再把结果返回给模型生成最终回复。对话历史存储在 MySQL 而不是内存里，因此服务重启后上下文依然保留。每个用户的对话通过 `user_id` 相互隔离。

## Features / 功能特点

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
- **DeepSeek API**（OpenAI 兼容接口） — the underlying LLM / 底层大模型
- **Pydantic** — request validation / 请求数据校验

## Architecture / 架构流程

```
User request 用户请求
    ↓
Save user message to MySQL 存入用户消息
    ↓
Load full conversation history for this user_id 读取该用户完整历史
    ↓
Send history + tool definitions to the LLM 发送历史+工具清单给大模型
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
pip install fastapi uvicorn pymysql openai
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

## Possible Extensions / 可扩展方向

- User authentication（目前 `user_id` 由调用方直接传入，没有正式的登录鉴权）
- Additional tools（网页搜索、天气 API 等）
- Streaming responses（流式返回）
- Rate limiting / usage tracking per user（限流与用量统计）
