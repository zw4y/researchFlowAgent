# 架构决策

## 模块化单体

FastAPI、LangGraph、LlamaIndex RAG 与 MCP 共用同一个应用服务层。模块边界通过 Python Protocol、Pydantic Schema 和服务类体现，避免第一版承担微服务通信、分布式事务和链路追踪成本。

## 数据职责

- SQLite：本地开发中的论文元数据、chunk、任务、会话、工具轨迹、引用和实验指标。
- PostgreSQL：后续部署时替换 SQLite，服务层接口不变。
- Qdrant Local：1024 维 chunk 向量与检索 payload，持久化在 `data/qdrant`。
- 上传目录：原始 PDF，重建索引时不需要重新上传。
- LangGraph State：单次运行的短期状态；conversation ID 关联持久化消息。

## RAG 分工

- PDF 解析器保留页码边界；含表格标题的页面使用 RapidOCR 补充单元格，并将数据集与 `Ours` 关键行压缩为优先检索摘要。
- LlamaIndex `SentenceSplitter` 在每页内生成节点，并维护节点 metadata。
- DashScope `text-embedding-v4` 分别以 `document` 和 `query` 模式生成向量。
- Qdrant 根据 `paper_id` 与 `index_profile` 过滤并召回 Top 20。
- `qwen3-rerank` 重排序为 Top 6；失败时退回向量顺序。
- RetrievalService 将 LlamaIndex 节点映射回稳定的 `Evidence` 契约。
- DeepSeek 只综合程序提供的证据，Citation Check 校验引用属于本次检索结果。

## 索引生命周期

`profile_id` 由 Provider、模型、维度、chunk size、overlap 和 splitter 版本生成。collection 名称包含 profile，禁止不兼容向量混用。

```text
pending/stale
→ indexing
→ 写入 profile collection
→ 更新 PostgreSQL/SQLite chunks
→ 原子切换 paper.index_profile
→ ready
```

失败时不激活不完整索引。应用启动只把不匹配当前 profile 的论文标记为 `stale`，不会自动调用付费 API。

## 信任边界

- LLM 输出永远视为不可信输入。
- 工具参数使用 Pydantic 校验。
- SQL 使用 AST 解析，只允许单条 `SELECT`、授权表与字段。
- `read_file` 只能访问受管上传目录中的文本、Markdown 和 CSV。
- 引用由程序从检索结果构造，模型不能伪造页码或 chunk ID。
- API Key 只从被 Git 忽略的 `.env` 读取，健康接口不返回密钥。

## 扩展点

- `ChatProvider`：切换 DeepSeek 或其他 OpenAI 兼容模型。
- `EmbeddingProvider`：生产环境使用百炼，测试通过容器构造参数注入离线替身。
- `RerankProvider`：切换重排序服务，并保留向量降级路径。
- `SearchProvider`：替换 Tavily。
- `LlamaIndexVectorStore`：从 Local Mode 切换远程 Qdrant。
- `ToolRegistry`：增加研究工具。

达到真实并发后，可将 `IngestionService` 移至任务队列 worker；API 和服务层契约无需改变。