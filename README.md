# Agent Backend

企业内部智能查询 Agent 后端项目。

项目目标：构建一个具备 **Tool Calling、Skill、RAG、Memory、MCP、权限控制、多用户隔离**能力的企业级 Agent 后端，并通过实际工程代码学习 Agent 的核心架构。

> 当前状态：核心 Agent 能力已经基本搭建完成，RAG 检索链路为「智能切分 → 结构化元数据 → 混合召回 → 重排 → 带引用回答」，Planner / TaskState 已完整接入 AgentLoop 并带失败降级与角色过滤，RAG 有独立评测集。最近一次全量验证（2026-10-19）**368 项，368 passed / 0 failed / 0 skipped / 5 warnings**（本机 Python 3.13 验证，`python -m pytest -q`；5 条警告均为 16 字节 `JWT_SECRET_KEY` 触发的 PyJWT 密钥长度警告，见第 10 节；仅 `test_main.py::test_register_new_user` 会写数据库，且验证使用一次性 MySQL 容器，未写开发库）。下一阶段重点是 Agent + Memory 深度结合与检索质量进阶（BM25 / Cross-Encoder）。

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

可选配置：

```text
MYSQL_HOST=localhost        # 默认 localhost，Docker 内部填 mysql
MYSQL_PORT=3306
MYSQL_USER=root
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password   # 可选，未设置时不做 AUTH
CORS_ALLOW_ORIGINS=https://your-frontend.example.com
AUTO_CREATE_TABLES=true     # 启动时自动补齐缺失表（不修改已有表）
LOG_FILE=logs/agent-backend.log
```

- 不设置 `CORS_ALLOW_ORIGINS` 时默认不开启跨域；多个来源用逗号分隔。
- `AUTO_CREATE_TABLES=false` 可关闭自动建表（有正式迁移流程的环境）。
- `REDIS_PASSWORD` 仅在 Redis 启用了 `requirepass` 时需要；留空或删除该行即按无密码连接。
- 设置 `LOG_FILE` 后日志同时写入轮转文件（5MB × 3），每行带 `request_id`，可用 `X-Request-ID` 请求头串联一次请求的所有日志。

依赖服务：MySQL（库名 `agent_db`）、Redis、DeepSeek API。

表结构默认**自动创建**：`AUTO_CREATE_TABLES=true`（代码默认值，无需在 `.env` 中显式设置）会在启动时补齐缺失的 `users` / `messages` / `conversation_summaries` 三张表，且不会修改已有表。只有设置 `AUTO_CREATE_TABLES=false` 的环境（例如已有正式迁移流程）才需要自行执行 DDL 建表。

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

基础 Tool 与 Skill 在 AgentLoop 中于**实际执行前**做程序侧权限校验，不依赖 Planner 隐藏工具。三类执行入口（基础 Tool、Skill、MCP）策略一致：`PermissionChecker` 缺失时**一律拒绝执行**（deny-by-default），不存在“没配 checker 就放行”的分支。

### MCP 工具授权（deny-by-default）

MCP 工具动态发现，因此采用显式授权映射 `PermissionChecker.MCP_ROLE_PERMISSIONS`：

```text
get_company_info → {employee, finance, admin}   # 静态公开信息
（未登记的任何 MCP 工具 → 所有角色默认拒绝）
```

- 当前仅显式登记 `get_company_info`（只返回静态公开公司资料，敏感度等同 `get_current_time`）。
- 其他 MCP 工具一律默认拒绝，不隐式继承任何角色权限；新增 MCP 工具必须显式登记后才可用。

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

### 限流与输入校验

| 端点 | 维度 | 预算 | 说明 |
| --- | --- | --- | --- |
| `POST /login` | 客户端 IP | 5 次 / 5 分钟 | **按 IP 计数，不包含用户名**：换用户名不会重置预算；登录成功后清零该 IP 计数 |
| `POST /register` | 客户端 IP | 5 次 / 1 小时 | 与登录使用各自独立的计数器 |
| `POST /chat` | 用户 | 30 次 / 分钟 | 在写入历史和调用 LLM **之前**返回 `429`，被拒请求不留半截数据 |
| `POST /upload_doc` | 用户 | 20 次 / 分钟 | 在权限校验与入库**之前**返回 `429` |

- 超限统一返回 `429`，并带 `Retry-After` 响应头。
- 注册输入校验：用户名 3–50 位且不含空白；密码 8–128 字节且不超过 bcrypt 的 72 字节上限。
- 计数器保存在**进程内存**中：单进程部署有效；多 worker / 多实例部署需要改用 Redis 等共享存储。
- 计数器内存有界：过期事件与闲置 key 会自动回收，key 数量超过上限（默认 1 万）时按 LRU 淘汰，大量唯一 key 无法无限撑大内存。

---

## 9. API

当前主要 API：

```text
POST /register
POST /login
POST /upload_doc
POST /chat
```

`POST /upload_doc` 支持可选 `title`，入库时记录 `uploaded_by`。

> **中文上传注意事项（Windows PowerShell 5.1）**
>
> PowerShell 5.1 的 `Invoke-RestMethod` 在 `-ContentType` 未声明
> `charset=utf-8` 时，会把请求体中的中文逐字符替换为字面量 `?`
> （请求尚未到达服务端即已损坏）。上传中文文档必须显式声明编码：
>
> ```powershell
> $json = '{"text":"接口测试内容","permission":"employee","title":"接口测试"}'
> Invoke-RestMethod -Uri "http://127.0.0.1:8000/upload_doc" `
>   -Method Post `
>   -Headers @{ Authorization = "Bearer $token" } `
>   -ContentType 'application/json; charset=utf-8' `
>   -Body $json
> ```
>
> 更稳妥的方式是显式构造 UTF-8 请求字节再发送，例如使用
> `curl.exe --data-binary "@utf8.json"`（文件须存为 UTF-8）或
> Python `requests.post(..., data=json.dumps(payload,
> ensure_ascii=False).encode("utf-8"), headers={"Content-Type":
> "application/json; charset=utf-8"})`。
>
> 注意：客户端丢失的中文无法由服务端恢复；请勿期望通过修改
> `ingest_document()` / `TextChunker` 修复已变成 `?` 的数据。

`POST /chat` 支持可选 `include_plan`：

```json
{"message": "帮我查一下差旅标准", "include_plan": true}
```

返回值额外包含 `plan`：

```json
{
  "reply": "...",
  "plan": {
    "status": "completed",
    "total": 2,
    "completed": 2,
    "failed": 0,
    "pending": 0,
    "running": 0,
    "steps": [
      {"description": "查询差旅标准", "tools": ["knowledge_search"], "status": "completed", "retry_count": 0}
    ]
  }
}
```

`status` 取值：`completed` / `in_progress` / `failed`；没有 Planner 时为 `null`。

---

## 10. 测试

当前全量测试：

```text
收集 368 项 —— 368 passed / 0 failed / 0 skipped / 5 warnings
```

以上为本机（Python 3.13）执行 `python -m pytest -q` 的验证结果（2026-10-19，耗时约 29s）。验证使用一次性 MySQL 8.4 容器与独立 Redis 容器，开发库零写入；5 条警告全部是 PyJWT 的 `InsecureKeyLengthWarning`（测试所用 `JWT_SECRET_KEY` 为 16 字节，低于 HS256 建议的 32 字节，属配置问题、非代码缺陷；`main.py` 在启动时也会打印一条只含长度的告警）。

其中仅 **1 项**会写数据库：`test_main.py::test_register_new_user` 通过 `/register` 追加测试用户（uuid 命名、只增不改不删），写入的是上述一次性容器。B7 隔离改造后，`tests/test_degradation.py` 与 `tests/test_upload_encoding.py` 均通过 `_StubDB` + `get_db` override 在内存中完成，不再触碰 MySQL。其余 367 项仅使用 Fake/Stub，不写库、不写向量库、不调用外部 LLM（`evals/run_rag_eval.py` 使用临时目录中的独立 Chroma collection）。

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
- Planner 失败降级、角色过滤、步数预算
- Retry / Recovery
- Error Handling
- 登录 / 注册限流与输入校验
- 登录限流按 IP 计数（换用户名无法绕过）、限流计数器过期回收与容量上限、`/chat` 与 `/upload_doc` 按用户限流（`tests/test_endpoint_rate_limits.py`）
- 问候识别回归：英文子串误判不再跳过 Planner（`tests/test_text.py`）
- 错误脱敏：裸 `sk-…` 密钥与 `/E:/…` 形式路径（`tests/test_error_sanitization.py`）
- Agent 并发请求状态隔离
- 注入防护：工具执行时权限校验、三类执行入口缺失 checker 时一律拒绝（`tests/test_rag_injection.py`）
- 摘要 / 计划信任边界标记（`tests/test_context_trust.py`）
- 输出层风险扫描与日志脱敏（`tests/test_output_guard.py`）

每次重要功能修改后都会执行完整 pytest，确保已有 Agent 能力没有被破坏。

### RAG 评测

```bash
python evals/run_rag_eval.py
```

```text
数据集   evals/rag_cases.jsonl   查询 + 角色 + 相关/禁止标题
语料     evals/corpus.jsonl      5 篇企业文档
指标     hit_rate / recall@k / MRR / no_result_rate / permission_leaks
```

评测会真实入库并检索，权限泄漏数大于 0 时进程返回非 0 退出码，可直接接进 CI。

当前指标：

```text
cases=8 top_k=3 hit_rate=1.0 recall@3=1.0 mrr=1.0 permission_leaks=0
```

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
Planner 完整接入           ✅
TaskState 状态回传         ✅
Planner 角色过滤 + 降级    ✅
RAG 评测集                 ✅
pytest                     ✅ 368/368 passed / 0 skipped / 5 warnings（JWT 密钥长度）
限流端点测试               ✅ 路由层 429 / Retry-After / 成功重置 / IP·用户隔离
X-Request-ID 校验          ✅ 白名单字符集 + 长度上限，非法值服务端重新生成
能力清单按角色过滤          ✅ 模型只看到本角色可执行的 tool/skill/MCP
数据库引导脱敏             ✅ 驱动缺失/DSN 错误 → 分类错误码，不回显 DSN
Compose healthcheck        ✅ MYSQL_PWD 传参，密码不进进程参数
脱敏边界测试               ✅ 明确"能脱什么/不能脱什么"，不承诺绝对不泄露
rag_eval                   ✅ hit_rate=1.0 leaks=0
登录限流 / 注册校验        ✅ 登录按 IP 计数，换用户名不可绕过
chat / upload 限流          ✅ 每用户 30/min、20/min，写库前拦截
限流计数器内存有界          ✅ 过期回收 + key 上限 LRU 淘汰
依赖故障降级               ✅ Redis / MCP / MySQL
并发状态隔离               ✅
输入长度限制               ✅
问候跳过 Planner           ✅ 整词匹配，普通英文句子不误判
历史消息限量               ✅
Tool / Skill / MCP 权限校验 ✅ deny-by-default（checker 缺失一律拒绝）
MCP 显式授权映射           ✅ MCP_ROLE_PERMISSIONS
错误脱敏扩展               ✅ 裸 sk-… 密钥、/E:/… 正斜杠路径
JWT 过期时间               ✅ 时区感知 UTC（不再用 utcnow）
RAG 不可信内容隔离         ✅ untrusted_document 边界
摘要 / 计划信任边界        ✅ 标记为对话生成上下文
输出层风险扫描             ✅ 仅告警，非防泄漏保证

Agent + Memory 深度结合    🚧
MCP 高级能力               ⏳
Observability              ✅ request_id 日志 / 轮转
Docker 部署                ✅ docker compose
CI                        ✅ pytest + rag_eval
```

---

## 12. 下一阶段路线

### Phase 1：完善 Planner（已完成）

执行链路：

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

重点（已完成）：

- Planner 真正参与 Agent 执行，`create_plan` 抛错时记录日志并降级为无计划模式，请求不会 500
- Planner 按角色过滤可用 Tool / Skill，无权使用的能力不会进入计划
- TaskState 记录任务状态与重试次数，Step 与 Tool / Skill 对应
- 支持任务成功 / 失败 / 重试，`summary()` 汇总整体状态
- `max_steps` 随计划长度自动放大，长计划不会被截断
- `/chat` 支持 `include_plan` 回传计划状态
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

## 13. 部署

### Docker Compose（推荐）

```bash
docker compose up -d --build
```

一键拉起 app + MySQL + Redis：

- `MYSQL_HOST=mysql`、`REDIS_HOST=redis` 由 compose 注入，无需改代码
- `AUTO_CREATE_TABLES=true` 启动时自动建表
- Chroma 向量库与日志通过 volume 持久化
- app 镜像以非 root 用户（UID 10001）运行
- compose 的 `mysql` 服务只创建 root 账号；app 默认 `MYSQL_USER=root`。如需普通账号，请在容器内创建并同步调整 `.env` 的 `MYSQL_USER` / `MYSQL_PASSWORD`

### CI

`.github/workflows/ci.yml`：每次 push / PR 自动起 MySQL + Redis 服务容器，执行 `pytest -q` 与 `python evals/run_rag_eval.py`（权限泄漏数 > 0 时 CI 失败）。

---

## 14. 技术栈

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

## 15. 项目定位

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
