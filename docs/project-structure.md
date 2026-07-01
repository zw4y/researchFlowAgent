# ResearchFlow Agent 项目结构

本文档说明项目中每个目录和主要代码文件的职责。

## 根目录

```text
researchflow-agent/
├── .github/               GitHub Actions 配置
├── backend/               Python 后端、Agent、RAG、MCP 和测试
├── data/                  本地运行数据，不提交 Git
├── demo/                  演示数据
├── docs/                  架构、学习路线和面试资料
├── frontend/              React 研究工作台
├── scripts/               本地启动与验收脚本
├── .env.example           环境变量模板
├── .gitignore             Git 忽略规则
├── alembic.ini            Alembic 迁移配置
├── compose.yaml           Docker Compose 编排
├── environment.yml        Conda 环境定义
├── pyproject.toml         Python 依赖和质量工具配置
└── README.md              项目入口文档
```

## 后端

```text
backend/
├── alembic/versions/0001_initial.py   初始数据库迁移
├── alembic/versions/0002_embedding_index_profile.py 生产索引生命周期迁移
├── app/agent/workflow.py              LangGraph 研究工作流
├── app/api/dependencies.py            FastAPI 依赖注入
├── app/api/routes.py                  HTTP 与 SSE 接口
├── app/core/config.py                 环境变量和应用配置
├── app/core/errors.py                 统一业务异常
├── app/core/logging.py                日志初始化
├── app/db/models.py                   SQLAlchemy 数据模型
├── app/db/session.py                  数据库引擎与会话
├── app/providers/base.py              LLM 与 Embedding 抽象
├── app/providers/dashscope.py         百炼 Embedding 与 Rerank 接入
├── app/providers/openai_compatible.py OpenAI 兼容模型接入
├── app/providers/search.py            Tavily 搜索接入
├── app/providers/tool_selector.py     Function Calling 工具选择
├── app/rag/pdf.py                     PDF 页级解析
├── app/rag/index_profile.py           索引版本与 collection 命名
├── app/rag/vector_store.py            LlamaIndex 与 Qdrant 向量存储
├── app/services/ingestion.py          论文摄取任务
├── app/services/metrics.py            实验指标导入和查询
├── app/services/papers.py             论文生命周期管理
├── app/services/retrieval.py          检索与引用组装
├── app/tools/registry.py              工具注册、执行和审计
├── app/container.py                   组装应用依赖
├── app/main.py                        FastAPI 应用入口
├── app/mcp_server.py                  MCP HTTP 与 stdio 入口
├── app/schemas.py                     API 和内部数据契约
├── tests/
│   ├── doubles.py                     仅供自动化测试注入的离线替身
│   ├── test_embedding_rag.py          Embedding、Rerank、Qdrant 与重建测试
│   └── test_*.py                      其他后端单元与集成测试
└── Dockerfile                         后端容器镜像
```

目录中的 `__init__.py` 是 Python 包标记。即使文件内容为空，也不代表功能未完成。

## 前端

```text
frontend/
├── src/test/setup.ts       Vitest 测试环境
├── src/api.ts              API 与 SSE 客户端
├── src/api.test.ts         API 客户端测试
├── src/App.tsx             研究工作台主界面
├── src/App.test.tsx        界面测试
├── src/main.tsx            React 启动入口
├── src/styles.css          响应式样式
├── src/types.ts            前后端数据类型
├── index.html              Vite HTML 入口
├── package.json            Node 依赖和命令
├── vite.config.ts          开发服务器与构建配置
├── vitest.config.ts        前端测试配置
├── nginx.conf              生产静态站点配置
└── Dockerfile              前端容器镜像
```

## 文档与脚本

- `docs/architecture.md`：系统架构、数据流和安全边界。
- `docs/interview.md`：项目讲解和面试问题。
- `docs/learning/`：按照技术依赖组织的 11 个学习阶段。
- `scripts/start-backend.ps1`：启动本地 FastAPI。
- `scripts/test-demo.ps1`：执行本地 Demo 验收。
- `demo/experiment_metrics.csv`：实验指标 CSV 示例。

## 本地生成目录

- `.venv/`：Python 虚拟环境；改用 Conda 后可以不再创建。
- `data/`：上传文件、SQLite、Qdrant Local 数据和测试产物，不提交 Git。
- `.pytest_cache/`、`.pytest-*/`：测试缓存，可以删除。
- `frontend/node_modules/`：npm 安装的依赖。
- `frontend/dist/`：前端生产构建结果。

在 PyCharm 中，目录左侧出现 `>` 表示目录处于折叠状态，并不表示它是空目录。
