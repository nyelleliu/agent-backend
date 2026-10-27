# Agent Backend

一个面向企业内部知识问答和受权限约束工具调用的 AI Agent 后端练习项目。项目用 FastAPI 提供 API，以 AgentLoop 编排模型、Tool、Skill、MCP 与 RAG，并用 MySQL、Redis 和 Chroma 保存业务状态、对话摘要及知识文档。

> **验证范围：** 本文描述当前 `main` 中的源码与评测文件，不代表本次 README 更新运行过测试、真实 LLM 评测或生产部署。离线 Fake Client 结果只代表预设响应下的流程检查；已记录的评测结果见“测试与评测”一节。

## 项目能力

- 多轮 AgentLoop、可选 Planner 和 TaskState 状态回传。
- Tool、Skill、MCP 统一以 schema 提供给模型，并在执行前按角色复核权限。
- RAG 文档切分、向量与关键词混合召回、重排、编号引用及文档级权限过滤。
- 用户级对话历史和摘要；Redis 缓存失败时从数据库回退。
- 差旅预算模拟工具，校验必填字段、城市、天数、职级和费用，并返回计算明细。
- 注册登录、JWT、速率限制、错误信息脱敏、请求 ID 和风险日志。

本项目是学习和面试演示用的后端原型，不是已验证的生产系统。差旅标准为模拟数据，不代表真实企业政策。

## 技术栈

Python 3.13、FastAPI、Pydantic、OpenAI Python SDK（连接 DeepSeek API）、SQLAlchemy、MySQL、Redis、Chroma、MCP Python SDK、PyJWT、bcrypt、pytest。

依赖版本见 [`requirements.txt`](requirements.txt)。

## 架构与执行流程

```text
HTTP request
    ↓
FastAPI authentication / rate limit / database user role
    ↓
MemoryManager loads user history
    ↓
API layer injects system prompt
    ↓
Planner (optional) creates a role-filtered plan
    ↓
AgentLoop offers role-filtered schemas to the model
    ↓
model tool call → execution-time permission check → Tool / Skill / MCP
    ↓
tool result returns to model → final reply + optional TaskState
    ↓
persist messages → best-effort summary compression → Redis cache
```

### AgentLoop、Planner 与 TaskState

`/chat` 将认证用户的角色和历史消息传给 AgentLoop。普通请求可先交给 Planner；Planner 只向模型提供当前角色获准的能力，解析并校验 JSON 计划，过滤未授权能力。若 Planner 请求超过预算，`ContextBudgetExceeded` 会使 AgentLoop 在后续模型调用前返回预算错误并终止当前请求；普通 Planner 异常或无效计划则记录降级并继续无计划流程。

计划作为引导上下文加入消息。AgentLoop 向 LLM 提交可用工具 schema；模型决定是否发起调用。AgentLoop 按工具名将调用映射到 TaskState 步骤，记录 running、completed、failed 与 retry_count，并把工具结果反馈给模型。最终答案和计划状态由 `/chat` 返回（通过 `include_plan` 请求状态）。TaskState 是执行过程状态记录，不是独立调度器，也不保证计划中的每一步都会被模型执行。

AgentLoop 默认最多 5 次模型循环；存在计划时循环上限会按计划长度放宽，并限制最多 12 次工具调用。Planner 和 AgentLoop 的应用实例都由 `main.py` 的 `APP_MAX_CONTEXT_CHARS` 接收同一个预算值，当前赋值为共享默认值 24,000 字符；这个常量不是读取 `.env` 的 `APP_MAX_CONTEXT_CHARS` 环境变量。组件单独实例化仍可传入 `max_context_chars`，小于 1,000 的值会按共享下限提升至 1,000。

预算按紧凑 JSON 请求体的字符数计量：`json.dumps(..., ensure_ascii=True, separators=(",", ":"), allow_nan=False)`。Planner 检查 `model` 与 `messages`；其能力 schema 嵌在 system message 中，没有独立 `tools` 字段。AgentLoop 检查 `model`、`messages` 和实际发送的 `tools` schema。AgentLoop 按完整消息结构裁剪较旧的可选历史组，保留 system 消息、当前用户请求和完整的 assistant tool call/tool result 组；畸形工具历史或必需上下文仍超限时，在模型请求前拒绝。Planner 超限也在其模型调用前抛出预算异常。此字符预算不等于模型 Token 数，也未通过供应商在线请求限制验证；它只覆盖上述 Planner 与 AgentLoop 请求路径。

### Tool、Skill、MCP 与权限

- Tool 通过 `ToolRegistry` 注册；当前默认工具包括 calculator、current time 和差旅预算。
- Skill 通过 `SkillRegistry` 注册；主要能力是 `knowledge_search` 和 `data_analysis`。
- MCP server 工具由 `MCPToolRegistry` 发现，经 Adapter 转成模型函数 schema，再由 MCPClient 通过 stdio 会话调用。内置示例 server 位于 `app/mcp_server.py`。
- AgentLoop 向模型展示能力前按角色过滤，并在每次执行前再次检查；未配置 PermissionChecker 时默认不展示、不执行。动态 MCP 工具使用单独的显式 allow-list，未列入的工具对所有角色拒绝。

当前规则概览：

| 角色 | 应用能力 | 可检索文档级别 | MCP |
|---|---|---|---|
| employee | knowledge_search、calculate、get_current_time、calculate_travel_budget | public、employee | `get_company_info` |
| finance | employee 能力及 data_analysis | public、employee、finance | `get_company_info` |
| admin | finance 级应用能力 | public、employee、finance、admin | `get_company_info` |

差旅工具限制为 employee、finance、admin 角色可用，但它只使用本地写定的模拟政策金额；当前没有真实报销系统集成。

`PermissionChecker` 对未知角色不给予 Tool/Skill 能力；动态发现的 MCP 工具必须出现在显式 allow-list 中，否则所有角色默认拒绝。缺少 AgentLoop 权限检查器时也拒绝执行。文档级别为 `public`、`employee`、`finance`、`admin`；上传接口通过枚举校验拒绝其他级别，上传者还必须有权写入该级别。文档检索先按角色允许级别过滤，并再次验证候选元数据；未知角色的文档权限回退仅为 `public`。

### 登录、限流与输入校验

注册使用 bcrypt 哈希密码；登录签发 HS256 JWT，令牌有效期 24 小时。认证后聊天与上传会从数据库读取当前用户角色，而不是依赖令牌中可能过期的角色值。

| 端点 | 计数范围 | 限额 |
|---|---|---|
| `/login` | 客户端 IP | 5 次 / 5 分钟；成功登录后重置该 IP 计数 |
| `/register` | 客户端 IP | 5 次 / 1 小时 |
| `/chat` | 用户 ID | 30 次 / 分钟 |
| `/upload_doc` | 用户 ID | 20 次 / 分钟 |

超限返回 `429` 和 `Retry-After`。限流器是有界的进程内滑动窗口；多 worker 或多实例之间不共享计数。注册用户名为 3–50 个非空白字符，密码为 8–128 个字符且 UTF-8 编码不超过 bcrypt 的 72 字节上限；聊天消息最多 8,000 字符，上传文本最多 100,000 字符、标题最多 200 字符。

## RAG 检索链路

1. `/upload_doc` 验证文档级别和上传者角色，然后切分文档并写入 Chroma 元数据。
2. `TextChunker` 按空行分段、按句子边界拆分，保留 overlap；chunk 元数据包括 permission、title、doc_id、chunk_index、source、uploaded_by 和 created_at。
3. Retriever 依次执行 Chroma 向量检索和关键词检索；两路都带文档权限过滤，并对返回元数据再次校验，再合并去重。
4. 默认使用启发式重排，也提供可选 LLM reranker（异常时回退启发式）。只有重排后的检索结果会被格式化为编号片段供 KnowledgeSearch Skill 返回。
5. 系统提示要求模型以 `[N]` 标注引用；被检索段落包在不可信内容边界中，并提示模型将其当作数据。

默认 `TextChunker` 使用 `chunk_size=500`、`overlap=80`，按段落和句子边界组织文本，超长单元再切分。每个 chunk 保存 `permission`、`title`、`doc_id`、`chunk_index`、`source`、`uploaded_by`、`created_at` 等元数据，ID 为文档 ID 与 chunk 序号组合。默认 `Retriever` 每路最多取 10 个候选、向量距离阈值为 1.5，关键词召回最多使用 8 个提取词；中文关键词提取包含二元词组。`KnowledgeSearchSkill` 将候选数设为 10，启发式重排后默认返回 3 段。Retriever 独立调用的 `retrieve(..., top_k=5)` 默认值不同于 Skill 路径。可选 `LLMReranker` 最多接收 10 个候选，失败时回退启发式排序。以上数值是各实现路径的默认值，可由构造参数覆盖。

权限在召回阶段执行，重排只处理已经过权限筛选的候选。提示词边界是降低提示注入风险的模型侧措施，不构成安全沙箱，也不证明攻击场景下绝无越权。输出守卫是尽力而为的日志监测：检测部分疑似密钥或注入文本模式，不拦截、不改写回答。

`evals/rag_cases.jsonl` 给出查询、角色、相关标题和禁止标题；评价代码实现 hit rate、recall@k、MRR、no-result rate 与 permission leak 计数。它评价检索结果与标题标签，不评价生成答案真实性或引用忠实度。数据集规模较小，需扩充后才能作为有代表性的质量结论。

## Memory 与用户数据范围

MemoryManager 读取最近最多 20 条消息；达到摘要阈值后将较早对话交给 LLM 压缩，并保存 summary 和覆盖的消息 ID。Redis 使用 `summary:{user_id}` 缓存，TTL 为 300 秒；缓存格式异常或 Redis 读写失败时使用数据库记录/继续数据库路径。

当前隔离范围是**用户级**：消息和摘要记录按 `user_id` 查询，缓存 key 也包含用户 ID。数据模型没有独立 conversation ID，因此不能声称同一用户的多个会话互相隔离。摘要提示要求总结事实和上下文，但没有单独保证保留全部权限约束；长期摘要保真仍需验证。

### Memory 摘要 schema 升级

应用的 ORM `create_all()` 只会创建缺失的表，不会升级已经存在的表或为旧表补列。升级旧版 `conversation_summaries` 前，先备份数据库，并确认要操作的 MySQL 主机和数据库名。迁移脚本不会读取 `.env`；必须在启动脚本的进程环境中显式设置非空 `MYSQL_HOST` 和 `MYSQL_DATABASE`，脚本不会替它们选择默认目标。`MYSQL_USER`、`MYSQL_PASSWORD` 和 `MYSQL_PORT` 也应按目标实例配置；用户和端口未设置时分别使用 `root` 和 `3306`。

Windows PowerShell 示例（先激活已有虚拟环境，并将主机、数据库及凭据替换为已核实的目标值）：

```powershell
.\.venv\Scripts\Activate.ps1
$env:MYSQL_HOST = "127.0.0.1"
$env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "root"
$env:MYSQL_PASSWORD = "<目标实例密码>"
$env:MYSQL_DATABASE = "agent_backend_dev"
python scripts/upgrade_memory_summary_state.py
```

迁移只添加缺失且符合契约的状态列；重复执行可继续已部分完成的升级。若失败，先根据错误类别检查目标主机/库、账号权限和 `conversation_summaries` 现有字段定义；不兼容字段需要人工评估，脚本不会自动重建、删除或转换字段。确认备份和目标无误、并处理阻塞原因后，可对同一目标重新运行上述命令。MySQL DDL 可能逐列隐式提交，因此失败不代表整组 DDL 已回滚。仓库中的 SQLite 测试不能证明 MySQL 的 DDL、锁等待或事务行为；这些行为尚未在真实 MySQL 实例验证。

## MCP 调用路径

应用组装时 `MCPToolRegistry.refresh()` 向 MCP Client 查询工具列表，再将发现的名称、描述和输入 schema 包装为 Adapter，供 AgentLoop 转成模型工具 schema。AgentLoop 在执行时按 MCP 专用 allow-list 再查权限；Registry 对未知工具返回错误，对调用异常记录脱敏错误类别并返回安全错误文本。当前 MCPClient 使用 stdio 会话和专用事件循环线程，默认启动与调用超时均为 30 秒；`main.py` 的 MCP 调用超时也显式设为 30 秒。应用导入时刷新失败会记录异常并继续启动，因此当时发现的 MCP 工具不可用；此降级路径不等于真实 MCP 服务已验收。

## 差旅预算工具

`calculate_travel_budget` 要求 destination city、days 和 job_level；支持上海、北京、深圳、广州，职级为 employee、senior、manager。天数限制为 1–90 的整数；金额要求有限、非负且不超过单项上限。工具按天计算住宿和餐补、按行程总额计算交通，返回政策预算、实际总额、超限项目及模拟数据说明。

可选参数为 `lodging_per_day`、`meal_per_day` 和 `transport_total`；未传时使用对应职级的模拟标准。上海、北京、深圳、广州共用同一组当前模拟值：employee 为 350/100/150，senior 为 500/150/250，manager 为 700/200/400（依次为住宿每日、餐补每日、整段行程交通总额）。必填字段缺失、未知参数、不支持城市或非法数值会拒绝并返回校验错误；每项费用最多 1,000,000，计算总额最多为 180,000,000。没有人数和日期输入，因此不会按人数缩放或根据日期调整。以上均为代码中的演示数据，不是公司制度金额。

该工具目前是确定性本地计算逻辑。`evals/run_agent_eval.py` 中的自然语言案例由 case ID 选择固定参数，不会解析 `message`；缺失字段等案例检查的是工具校验路径。`evals/run_agent_loop_eval.py` 通过 Fake Client 将预设模型响应走过真实 AgentLoop 与工具注册流程；多轮更正场景能检查脚本给定的新参数是否执行正确，但不能证明真实模型能理解更正语句。

## API

| Endpoint | 用途 | 认证 |
|---|---|---|
| `POST /register` | 注册用户 | 否 |
| `POST /login` | 登录并签发 JWT | 否 |
| `POST /upload_doc` | 上传带权限级别的知识文档 | Bearer JWT |
| `POST /chat` | 对话、工具调用和可选计划状态 | Bearer JWT |

登录返回 `access_token`。`/chat` 请求体含 `message`，可选 `include_plan: true`；`/upload_doc` 接受 `text`、`permission` 和可选 `title`。登录、注册、聊天和上传有速率限制。`/chat` 的角色从数据库用户记录读取，不信任请求体自报角色。

## 本地运行

需要 Python 3.13、MySQL 和 Redis。应用默认连接本机 MySQL `agent_db` 与 Redis；LLM 请求使用 DeepSeek API。Chroma 使用本地 `./chroma_data` 持久化目录。

### Windows 本地 Python

在根目录创建本地 `.env`（该文件已被 `.gitignore` 忽略），用于 Windows 本地 Python 启动。设置以下变量。**填写本机真实值，不要把密钥提交到 Git：**

```dotenv
DEEPSEEK_API_KEY=<your-provider-key>
MYSQL_PASSWORD=<your-local-mysql-password>
JWT_SECRET_KEY=<random-secret-at-least-32-bytes>
MYSQL_USER=root
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=agent_backend_dev
REDIS_HOST=localhost
REDIS_PORT=6379
# 可选：REDIS_PASSWORD=<your-local-redis-password>
AUTO_CREATE_TABLES=true
CHROMA_PERSIST_DIRECTORY=D:/agent-backend-dev-data/chroma
MCP_AUTO_REFRESH=false
```

| 配置项 | 默认值 | 作用 |
|---|---|---|
| `MYSQL_DATABASE` | `agent_db` | 指定 MySQL 数据库名。改名不代表目标数据库已经安全；必须确认它是专用开发数据库。Compose 会把同一个值传给应用和 MySQL 服务。 |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_data` | 指定 Chroma 持久化目录；相对路径按进程当前工作目录解析。Windows 本地开发建议使用仓库原有 `chroma_data` 之外的独立目录。 |
| `MCP_AUTO_REFRESH` | `false` | 控制应用导入时是否自动刷新 MCP Registry。值需为明确布尔值（`true`/`false`、`1`/`0`、`yes`/`no` 或 `on`/`off`）。 |

关闭 `MCP_AUTO_REFRESH` 只关闭导入时的自动发现，不会删除 MCP Client、Registry 或 Adapter；需要使用 MCP 时显式设为 `true`。`MYSQL_USER`、`MYSQL_HOST`、`MYSQL_PORT`、`REDIS_HOST`、`REDIS_PORT` 有本地默认值；`DEEPSEEK_API_KEY`、`MYSQL_PASSWORD` 和 `JWT_SECRET_KEY` 是启动所需配置。其他可选项包括 `REDIS_PASSWORD`、`CORS_ALLOW_ORIGINS`、`LOG_LEVEL`、`LOG_FILE` 及 `AUTO_CREATE_TABLES`。

`AUTO_CREATE_TABLES=false` 只跳过应用导入阶段的自动建表，不隔离数据库，也不阻止后续 API 请求读写数据库。只有确认 MySQL 指向专用开发库后，才应决定是否启用自动建表；禁用时需自行准备兼容的 schema。

### Docker Compose

Compose 对应用容器和 MySQL 初始化配置使用同一个 `${MYSQL_DATABASE:-agent_db}` 插值结果；通常从根目录 `.env` 读取，未设置时两边均为 `agent_db`。若改用其他数据库名，两边仍保持一致。`AUTO_CREATE_TABLES` 仍控制应用启动时是否建表，不提供数据库隔离。

Compose 会在应用容器环境中将 `CHROMA_PERSIST_DIRECTORY` 固定为 `/app/chroma_data`，并由 `chroma_data` 命名卷挂载持久化。它会覆盖 `.env` 中给 Windows 本地 Python 使用的 Chroma 路径；不要把 Windows 路径传给容器。`MCP_AUTO_REFRESH` 默认关闭，需要应用启动时发现 MCP 工具时再显式设为 `true`。

```bash
python -m venv .venv
# 激活虚拟环境后：
pip install -r requirements.txt
uvicorn main:app --reload
```

若使用 Docker Compose，可在配置好 `.env` 后运行：

```bash
docker compose up --build
```

仓库提供 `Dockerfile` 与 `docker-compose.yml`，Compose 配置会启动 API、MySQL 和 Redis。Compose 使用 `.env`，其中 `MYSQL_PASSWORD` 必须设置。运行或部署前请自行检查数据卷、网络、密钥管理和环境隔离；本仓库代码存在不等于生产部署已验证。

Compose 将 API 暴露在主机 `8000` 端口，MySQL 与 Redis 仅在 Compose 网络内供应用连接；`mysql_data`、`redis_data`、`chroma_data` 和 `app_logs` 为命名卷。MySQL healthcheck 等待数据库可用，API 依赖 MySQL healthy 和 Redis started。Dockerfile 以 UID 10001 的 `appuser` 运行 API。配置中的 MySQL 初始账号为 root，部署前应评估并按需改用权限受限账号；这些配置是本地/容器启动基线，不代表生产安全审查已完成。

## 测试与评测

运行测试：

```bash
pytest -q
```

`.github/workflows/ci.yml` 在 push 和 pull request 时使用 Python 3.13，启动 MySQL 8.4 与 Redis 7，安装 `requirements.txt`，运行 `pytest -q` 和 `python evals/run_rag_eval.py`。这是 CI 工作流定义，不表示当前工作区对应提交已通过 CI。当前 `main` 已包含根目录 pytest `conftest.py`：它在 pytest 收集阶段把应用导入接到临时 SQLite 和离线替身。该隔离由 pytest 配置提供，不会自动应用于独立运行的脚本；运行独立评测时仍需分别检查其调用路径和外部依赖。

运行 RAG 本地检索评测（使用临时 Chroma collection）：

```bash
python evals/run_rag_eval.py
```

该评测不调用生成式 LLM，默认使用启发式重排；Chroma 的默认 embedding function 依赖的模型是否需要首次下载，取决于本机缓存和安装版本，不能仅凭脚本本身保证完全离线。

运行 AgentLoop Fake Client 离线场景：

```bash
python evals/run_agent_loop_eval.py
```

该 AgentLoop 脚本默认离线；真实模型请求只在显式传入 `--live` 且提供相应环境变量时启用。Live 评测会发起外部网络 API 请求并可能产生费用，运行前应检查案例、预算和输出选项。此前 F2 验收记录中，AgentLoop Fake 评测的 12 个场景全部通过（12/12），Fake Client 共记录 29 次模型调用尝试，工具实际执行 11 次；这些是预设响应下的流程指标，不是自然语言理解成功率。Fake Client 的 usage 也不等于真实 API token 用量或费用。差旅工具契约评测共 20 案，18 passed、2 failed；已知失败 `correct_city`、`correct_days` 保持失败，runner 预期退出码为 1。这是工具契约检查，不是自然语言模型评测。本 README 修订过程中没有运行 pytest、RAG 评测或真实模型评测，因此不报告本轮新跑出的结果。仓库中的测试文件是覆盖意图，不等同于测试通过记录；CI 配置声明了 pytest 和 RAG 评测步骤，也不代表本次提交已经通过 CI。

`evals/run_agent_eval.py` 中的 `correct_city` 和 `correct_days` 是当前尚未实现的更正场景；案例标记为 `not_implemented`，但 runner 仍将它们计为失败，并会因此返回非零退出码。判读该评测时，应把常规工具契约验收与这两项已知功能缺口区分开；它们不是通过的功能验收，不能通过降低断言或伪造通过来消除红项。该 runner 根据案例 ID 选择固定参数，不解析自然语言，因此结果不代表自然语言理解能力。

## 安全与当前边界

- 不要提交 `.env`、API Key、MySQL/Redis 密码或 JWT 签名密钥。使用环境变量或本机 `.env`；生产环境应使用专门的 Secret 管理服务。
- `JWT_SECRET_KEY` 应为足够长的随机值；代码只记录密钥长度告警，不应将密钥值写入日志。
- Tool/Skill/MCP 的权限校验在 AgentLoop 执行层再次完成；RAG 文档权限过滤在两路召回中执行。提示词和输出扫描不是权限边界的替代品。
- 用户消息、摘要和检索文档是不可信输入；摘要与 RAG 注入防护仍需要对抗性场景验证。
- 请求预算限制的是特定 Planner/AgentLoop 路径上的序列化字符数，不是 Token 计数，也没有供应商在线验证；不能据此推断其他模型请求路径均已受保护。
- 项目没有按 conversation ID 隔离历史；MCP 目前演示 stdio 集成，尚无多服务器编排或生产 MCP 服务验证。
- 差旅标准是模拟数据；没有接入公司制度源或真实报销 API。
- Agent 与 Memory 有持久化和摘要路径，但还没有基于真实模型的长期记忆质量评测。

## 后续计划

1. 对用户级与会话级 Memory 边界作产品决定；如支持多会话，增加 conversation ID 和隔离测试。
2. 扩充标注语料和 RAG 查询集，单独评估生成答案与引用质量。
3. 在受控预算和脱敏条件下补充真实模型评测；清楚区分在线模型结果与 Fake Client 离线结果。
4. 按需完善 MCP 多 server、幂等工具、副作用控制和部署验证。
