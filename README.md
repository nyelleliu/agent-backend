# Agent Backend

企业内部智能查询 Agent 后端项目。

项目目标：构建一个具备 **Tool Calling、Skill、RAG、Memory、MCP、权限控制、多用户隔离**能力的企业级 Agent 后端，并通过实际工程代码学习 Agent 的核心架构。

> 当前状态：核心 Agent 能力已经基本搭建完成，RAG 检索链路已升级为「智能切分 → 结构化元数据 → 混合召回 → 重排 → 带引用回答」，当前全量测试 **105 passed**。下一阶段重点是把 Planner / TaskState 完整接入 AgentLoop。

---

## 0. 快速开始

```bash
pip install -r requirements.txt
```

`.env` 必需配置：

```text
DEEPSEEK_API_KEY=...
MYSQL_PASSWORD=...
JWT_SECRET_KEY=...
LOG_LEVEL=INFO
```

依赖服务：MySQL（库名 `agent_db`，需自建 `users` / `messages` / `conversation_summaries` 三张表）、Redis、DeepSeek API。

```bash
uvicorn main:app --reload
pytest -q
```

---

## 1. 项目架构

```text
                         User
                           │
                           ▼
                        FastAPI
                           │
                           ▼
                      AgentLoop
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Planner         Skill Registry    MCP Registry
          │                │                │
       TaskState           │                │
          │                ▼                ▼
          │             Tool Registry    MCP Adapter
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                  Permission Checker
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             Tool         Skill         MCP
              │            │            │
              └────────────┼────────────┘
                           ▼
                          LLM
                           │
                           ▼
                         Memory
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
                MySQL             Redis
```

---

## 2. 已完成模块

### Agent Core

- Agent Loop
- 多轮 Tool Calling
- Tool Registry
- Skill Registry
- Tool 参数处理
- Tool 错误反馈
- Retry / Recovery 基础能力
- Permission Checker
- Task State / Planner 基础代码

### Tools

当前已经接入：

- Calculator
- Current Time
- Tool 统一注册
- Tool schema
- Tool 参数处理与执行

### Skills

当前主要 Skill：

- Knowledge Search
- Data Analysis

Skill 可以通过统一 Registry 被 Agent 调用，并受到角色权限控制。

---

## 3. 权限系统

项目采用统一的 `PermissionChecker` 管理角色权限和文档权限。

### Role → Skill

```text
employee
├── knowledge_search
├── calculate
└── get_current_time

finance
├── knowledge_search
├── calculate
├── get_current_time
└── data_analysis

admin
├── knowledge_search
├── calculate
├── get_current_time
└── data_analysis
```

### Document Permission

```text
public
employee
finance
admin
```

文档上传权限已经统一使用：

```text
PermissionChecker.DOCUMENT_PERMISSIONS
```

避免 Knowledge Search 和 Upload Document 各自维护重复权限规则。

同时已经增加非法文档权限校验，例如：

```text
permission = "invalid"
        ↓
拒绝
```

---

## 4. RAG

RAG 检索链路位于 `app/rag/`：

```text
Document
   ↓  TextChunker（标题/段落/句子切分 + overlap）
Chunk
   ↓  ingest_document（结构化元数据）
Chroma
   ↓
Vector Query ──┐   Keyword Query（$contains / $or）
   ↓           │      ↓
 distance 阈值 │      词命中率打分
   └───── 合并去重 ─────┘
   ↓
 Reranker（启发式 / 可选 LLM）
   ↓
 Top-K 编号片段
   ↓
 LLM 按 [N] 引用作答
```

### 切分与元数据

```python
TextChunker(chunk_size=500, overlap=80)
```

- 先按 Markdown 标题 / 空行分段，再按中英文句子切分，超长段落硬切并保留重叠
- 每个 chunk 保存 `permission`、`title`、`doc_id`、`chunk_index`、`source`、`uploaded_by`、`created_at`
- chunk id 形如 `{doc_id}:{index}`，支持按文档删除与重建

### 召回配置

```python
n_results = 10            # 候选集，交给重排
distance_threshold = 1.5  # 向量腿低相关过滤
keyword_limit = 8         # 关键词腿最多 8 个词（中文取 2-gram）
top_k = 3                 # 重排后进入 LLM 的片段数
```

两路结果按 id 合并去重，双路命中会额外加分。

### 重排与引用

- 默认 `HeuristicReranker`：词面重合 + 标题加权 + 原始召回分，本地零成本
- 可选 `LLMReranker`：用 DeepSeek 给候选排序，失败自动回退启发式
- `knowledge_search` 返回编号片段：

```text
[1] 差旅报销标准 (permission=employee, score=0.61)
...
```

System Prompt 会要求模型按 `[N]` 标注出处。

### 当前状态

```text
文档切分              ✅ 标题/段落/句子 + overlap
结构化元数据          ✅
Embedding             ✅
Chroma                ✅
向量 + 关键词混合召回 ✅
Top-K                 ✅
Distance Filtering    ✅
文档上传权限          ✅
RAG 文档级检索权限    ✅
重排                  ✅ 启发式 / 可选 LLM
答案引用来源          ✅
```

---

## 5. MCP

已经完成 Agent 与 MCP 的基础集成。

当前架构：

```text
AgentLoop
    ↓
MCPToolRegistry
    ↓
MCPToolAdapter
    ↓
MCPClient
    ↓
Thread
    ↓
Event Loop
    ↓
MCP Server
```

MCP Client 已处理：

- MCP Server 启动
- 独立 Thread
- Async Event Loop
- `ClientSession`
- `initialize()`
- `list_tools()`
- `call_tool()`
- Session 生命周期管理

MCP Tool 会通过 Adapter 转换成 AgentLoop 可以统一调用的 Tool。

---

## 6. Memory

已经实现持久化对话 Memory。

当前包括：

- 多轮对话历史
- Conversation Summary
- 长对话自动压缩
- 数据库保存 Summary
- Redis Summary Cache

基本流程：

```text
Conversation
     ↓
History
     ↓
达到压缩阈值
     ↓
Summary
     ↓
ConversationSummary
     ↓
Redis Cache
```

Redis Summary Key：

```text
summary:{user_id}
```

并设置 TTL。

---

## 7. 多用户隔离

项目支持基于 `user_id` 的用户数据隔离。

主要应用于：

- Conversation History
- Memory Summary
- Redis Cache

目标是让不同用户之间的 Agent 上下文和历史数据相互隔离。

---

## 8. Authentication

项目已经包含基础用户认证能力：

- 注册
- 登录
- bcrypt 密码哈希
- JWT
- 用户角色

主要 API：

```text
POST /register
POST /login
```

---

## 9. API

当前主要 API：

```text
POST /register
POST /login
POST /upload_doc
POST /chat
```

---

## 10. 测试

当前全量测试：

```text
105 passed
```

测试覆盖主要包括：

- Tools
- Tool Registry
- Agent Loop
- Skills
- Permission
- Document Upload Permission
- Memory
- MCP
- RAG Chunker / Ingest / Retriever / Reranker
- Knowledge Search 引用与权限
- Planner / Task State 相关基础能力
- Retry / Recovery
- Error Handling

每次重要功能修改后都会执行完整 pytest，确保已有 Agent 能力没有被破坏。

---

## 11. 当前项目状态

```text
Tool Registry              ✅
Skill Registry             ✅
Agent Loop                 ✅
Tool Calling               ✅
Role → Skill Permission    ✅
Document Upload Permission ✅
RAG 智能切分 + 元数据      ✅
RAG 混合召回               ✅
RAG 重排 + 引用来源        ✅
RAG 文档级检索权限         ✅
Memory                     ✅
Redis                      ✅
Multi-user                 ✅
Authentication             ✅
MCP 基础集成               ✅
pytest                     ✅ 105 passed

Planner 完整接入           🚧
TaskState 完整接入         🚧
Planner → Tool/Skill       🚧
Agent + Memory 深度结合    🚧
MCP 高级能力               ⏳
Evaluation                 ⏳
Observability              ⏳
Docker 部署                ⏳
```

---

## 12. 下一阶段路线

### Phase 1：完善 Planner

将 Planner 和 TaskState 真正接入 AgentLoop：

```text
User Task
   ↓
Planner
   ↓
TaskState
   ↓
Step 1
   ↓
Tool / Skill
   ↓
Result
   ↓
TaskState Update
   ↓
Step 2
   ↓
...
   ↓
Final Answer
```

重点：

- Planner 真正参与 Agent 执行
- TaskState 记录任务状态
- Step 与 Tool / Skill 对应
- 支持任务成功 / 失败 / 重试
- 增加完整集成测试

### Phase 2：RAG 质量（已完成）

已完成：

- 检索阶段的文档权限过滤
- 智能切分（标题 / 段落 / 句子 + overlap）与结构化元数据
- 向量 + 关键词混合召回与去重融合
- 启发式重排，可切换 LLM 重排
- 带编号引用的回答与 Prompt 约束

```text
employee
    ↓
public + employee

finance
    ↓
public + employee + finance

admin
    ↓
全部
```

后续可继续：

- 接入 Cross-Encoder 或专用 Embedding Rerank 模型
- 中文关键词召回升级为 BM25
- RAG 检索质量评测集

### Phase 3：Agent + Memory

进一步让 Memory 深度进入 Agent：

```text
User
 ↓
Memory
 ↓
Agent
 ↓
Tool / Skill / RAG / MCP
 ↓
Result
 ↓
Memory
```

### Phase 4：MCP 高级能力

继续完善：

- 动态 Tool Discovery
- MCP Tool 生命周期
- 多 MCP Server
- Agent 动态扩展外部工具

### Phase 5：工程化

后续继续加入：

- Agent Evaluation
- Logging / Observability
- Docker
- 部署
- 更完整的集成测试

---

## 13. 技术栈

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
- MCP

---

## 14. 项目定位

最终目标：

> 构建一个面向企业内部场景的 AI Agent 后端，能够根据用户权限访问企业知识库、调用工具、分析数据、使用外部 MCP 工具，并通过 Memory 支持持续的多轮任务。

本项目同时用于系统学习：

- Agent Loop
- Tool Calling
- Skill Architecture
- Planner
- Task State
- RAG
- Memory
- MCP
- Permission System
- Agent Backend Engineering

项目重点不是简单调用 LLM API，而是理解并实现一个 **可扩展、可测试、具有权限控制和工具编排能力的 Agent 后端架构**。
