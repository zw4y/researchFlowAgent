# ResearchFlow Agent

> 项目目录与文件职责说明：[docs/project-structure.md](docs/project-structure.md)

面向论文与技术资料的可溯源研究工作流 Agent。系统能够摄取 PDF、检索页码级证据、重排序候选片段、查询实验指标、记录工具轨迹，并通过 LangGraph、HTTP API 和 MCP 复用同一套研究能力。

## 当前技术栈

- DeepSeek：对话、意图路由、Function Calling 与答案综合。
- 阿里百炼 `text-embedding-v4`：文档和查询向量，固定 1024 维。
- 阿里百炼 `qwen3-rerank`：向量召回 Top 20 后重排序为 Top 6。
- LlamaIndex：`Document`、`SentenceSplitter`、`VectorStoreIndex` 与 Qdrant 节点管理。
- RapidOCR + PDFium：在普通文本提取缺失表格单元格时补充表格结构与数值。
- Qdrant Local：向量磁盘持久化，不需要 Docker。
- LangGraph：研究工作流编排；FastAPI：HTTP/SSE 接口。
- SQLite：本地开发元数据；PostgreSQL 保留为部署形态。

## RAG 数据流

```text
PDF
→ 按页提取文本
→ 检测表格页并执行 RapidOCR，生成数据集/模型关键行摘要
→ LlamaIndex SentenceSplitter（800 tokens / overlap 120）
→ text-embedding-v4 文档向量（1024d）
→ Qdrant Local 持久化

用户问题
→ text-embedding-v4 查询向量（query + 研究检索指令）
→ Qdrant metadata filter
→ Top 20
→ qwen3-rerank
→ Top 6
→ DeepSeek 综合回答
→ Citation Check
```

每条论文引用保留 `paper_id`、论文标题、页码、`chunk_id`、证据片段和分数。Rerank 不可用时系统退化为向量排序，并在工具轨迹中记录 `rerank_fallback`。

## 本地启动（Conda）

```powershell
cd C:\Users\lenovo\Documents\agent开发\researchflow-agent
conda activate researchflow
python -m pip install -e ".[dev]"
python -m alembic upgrade head
python -m uvicorn app.main:app --app-dir backend --reload
```

另开一个已激活 Conda 的终端：

```powershell
cd C:\Users\lenovo\Documents\agent开发\researchflow-agent\frontend
npm install
npm run dev
```

打开 `http://localhost:5173`，API 文档位于 `http://localhost:8000/docs`。

## 模型配置

从模板创建本地配置：

```powershell
Copy-Item .env.example .env
```

`.env` 已被 Git 忽略。真实密钥只能放在 `.env`，不能写入 `.env.example`、源码、截图或提交记录。

正式本地检索配置：

```dotenv
LLM_MODE=openai_compatible
CHAT_API_KEY=
CHAT_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-flash
CHAT_THINKING=disabled

EMBEDDING_MODE=dashscope
DASHSCOPE_API_KEY=
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024

RERANK_MODE=dashscope
RERANK_MODEL=qwen3-rerank
RETRIEVAL_CANDIDATES=20
RERANK_TOP_N=6

VECTOR_STORE_MODE=qdrant_local
QDRANT_PATH=./data/qdrant
```

生产运行只使用 DeepSeek 与百炼。自动化测试通过 `backend/tests/doubles.py` 注入离线测试替身，不会读取真实密钥、访问外部 API 或产生费用。

缺少模型密钥时应用仍可启动并返回降级健康状态；上传和重建索引接口会明确返回 `503 embedding_not_configured`。

## 索引版本与重建

索引 `profile_id` 由 Provider、模型、维度、chunk size、overlap、splitter 与表格 OCR 配置共同生成。不同 profile 使用不同 Qdrant collection，避免测试索引、旧模型索引与 1024 维生产向量混用。

升级后，旧测试索引会显示为“索引过期”：

1. 启动后端和前端。
2. 在左侧论文列表点击“重建论文索引”按钮。
3. 等待状态变为“可检索”。
4. 只有当前 profile 为 `ready` 的论文可以加入问答。

系统先写入新 collection 和新 chunk；全部成功后才切换论文的 active profile。启动时只检测过期状态，不会自动调用付费 API。

## 核心接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/v1/health` | 模型、数据库和向量服务状态 |
| GET | `/api/v1/index/status` | 当前模型、维度、profile、向量数和论文索引统计 |
| POST | `/api/v1/papers` | 上传 PDF，返回论文与摄取任务 ID |
| GET | `/api/v1/papers` | 查询论文及索引状态 |
| POST | `/api/v1/papers/{id}/reindex` | 异步重建单篇论文索引 |
| GET | `/api/v1/ingestion-jobs/{id}` | 查询摄取或重建进度 |
| POST | `/api/v1/chat` | 完整研究回答 |
| POST | `/api/v1/chat/stream` | SSE 流式研究回答 |
| POST | `/api/v1/papers/{id}/metrics/import` | 导入实验指标 CSV |
| GET | `/api/v1/conversations/{id}` | 会话和工具轨迹 |
| POST | `/mcp` | MCP Streamable HTTP |

## 数据迁移

```powershell
python -m alembic current
python -m alembic upgrade head
```

迁移 `0002_embedding_index_profile` 增加论文索引状态、active profile、索引时间、chunk profile，以及摄取任务类型和详情。已有 PDF 和数据库记录会保留。

## 质量检查

```powershell
python -m ruff check backend
python -m mypy backend/app
python -m pytest backend/tests

cd frontend
npx tsc -b --pretty false
npm run lint
npm test
npm run build
```

真实百炼调用不进入默认 CI；默认测试通过依赖注入的离线替身与 mock 验证文档/查询参数差异、批处理、维度校验、Rerank、降级、profile 变化、Qdrant Local 持久化和重建 API。

## MCP

HTTP 客户端连接 `http://localhost:8000/mcp`。stdio 入口为：

```powershell
researchflow-mcp
```

MCP 工具包括 `search_papers`、`ask_paper`、`query_experiment_metrics` 和 `get_citations`。

## 学习路线

从 [docs/learning/README.md](docs/learning/README.md) 开始。建议依次理解配置与 Provider、PDF 摄取、LlamaIndex 节点、Qdrant filter、Rerank 降级、LangGraph 编排和引用校验。
