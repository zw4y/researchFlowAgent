# 架构决策

## 模块化单体

FastAPI、LangGraph、RAG、MCP 共用同一个应用服务层。模块边界通过 Python Protocol、Pydantic Schema 和服务类体现，避免第一版承担微服务通信、分布式事务和链路追踪成本。

## 数据职责

- PostgreSQL：论文元数据、chunk 原文、任务、会话、Agent Run、工具轨迹、引用和实验指标。
- Qdrant：chunk 向量与检索 payload。
- 上传卷：原始 PDF。
- LangGraph State：单次运行期间的短期状态。
- conversation ID：跨请求会话标识，消息持久化在数据库。

## 信任边界

- LLM 输出永远视为不可信输入。
- 工具参数使用 Pydantic 校验。
- SQL 使用 AST 解析，只允许单表、单条 `SELECT`、授权字段与函数。
- `read_file` 只能访问受管目录中的文本、Markdown 和 CSV。
- 引用由程序从检索结果构造，模型不能伪造页码或 chunk ID。

## 扩展点

- `ChatProvider`：切换模型供应商。
- `EmbeddingProvider`：切换 Embedding 服务。
- `SearchProvider`：替换 Tavily。
- `VectorStore`：内存与 Qdrant。
- `ToolRegistry`：增加研究工具。

达到真实并发后，可将 `IngestionService` 移至 Celery/Arq worker；模块接口不变，API 仅改为投递任务。

