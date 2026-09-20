# Agent Backend

企业内部智能查询 Agent 后端项目。

项目目标：构建一个具备 Tool Calling、Skill、RAG、Memory、Planner、权限控制和任务恢复能力的企业级 Agent。

## 当前架构

```text
User
 ↓
FastAPI
 ↓
AgentLoop
 ├── Planner
 ├── TaskState
 ├── Tool Registry
 ├── Skill Registry
 ├── Permission Checker
 └── Retry / Recovery
      ↓
   Tool / Skill
      ↓
   Result / Error
      ↓
   LLM
```

## 已完成

### Agent Core
- Agent Loop
- 多轮 Tool Calling
- Tool Registry
- Skill Registry
- Planner
- Task State
- Retry / Recovery
- Tool Error Feedback

### Tools
- Calculator
- Current Time
- Tool 参数校验
- Tool 统一注册与执行

### Skills
- Knowledge Search
- Data Analysis
- Skill 权限控制

### RAG
- Chroma 向量数据库
- 文档切分
- Embedding 检索
- Top-K 查询
- Distance Threshold 过滤

> RAG 文档级权限过滤尚未实现。

### Memory
- MySQL 保存对话
- Redis 缓存 Summary
- 长对话自动压缩
- Conversation Summary

### Authentication
- 注册 / 登录
- bcrypt 密码哈希
- JWT
- 用户角色

### Planner / Recovery

```text
用户任务
 ↓
Planner
 ↓
TaskState
 ↓
执行 Tool / Skill
 ↓
成功 → completed
失败 → failed
       ↓
     retry
       ↓
    pending
       ↓
   再次执行
```

Tool 失败时会将错误原因反馈给 LLM，使 Agent 能根据错误决定后续操作。

## API

```text
POST /register
POST /login
POST /upload_doc
POST /chat
```

## 技术栈

- Python
- FastAPI
- DeepSeek API
- SQLAlchemy
- MySQL
- Redis
- Chroma
- JWT
- bcrypt
- pytest

## 测试

当前测试：

```text
39 passed
```

测试覆盖：

- Tools
- Tool Registry
- Agent Loop
- Skills
- 权限
- Memory
- Planner
- TaskState
- Agent Planner Integration
- Retry / Recovery
- Error Feedback

## 项目路线

```text
Tool Registry        ✅
Agent Loop           ✅
Skill Registry       ✅
Role → Skill 权限    ✅
Memory               ✅
多轮 Tool Call       ✅
Planner              ✅
Task State           ✅
Retry / Recovery     ✅
Error Feedback       ✅
测试                  ✅

RAG 文档级权限过滤   ⏳
MCP                   ⏳
Multi-Agent           ⏳
Evaluation            ⏳
Logging               ⏳
Docker                ⏳
```

## 后续目标

1. MCP
2. RAG 文档级权限过滤
3. Multi-Agent
4. Agent Evaluation
5. Logging / Observability
6. Docker 部署

## 项目定位

最终目标：

> 企业级 AI Agent 助手：能够查询企业知识、调用业务工具、分析数据，并根据用户权限完成多步骤任务。

本项目同时用于学习 Agent 原理、Agent Framework 架构以及企业级 Agent 后端工程实践。
