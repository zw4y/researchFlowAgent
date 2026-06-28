# 01 工程骨架

## 学习目标

理解 Python 包、配置、依赖注入、异步数据库会话、日志、环境变量和前后端目录边界。

## 代码地图

- `backend/app/core`：Settings、错误与 JSON 日志。
- `backend/app/db`：SQLAlchemy 模型和异步 Session。
- `backend/app/container.py`：组合所有依赖。
- `pyproject.toml`：Python 依赖与质量工具。

## 动手

启动 API，访问 `/api/v1/health` 和 `/docs`；把 `LLM_MODE` 从 `fake` 改成 `openai_compatible`，观察健康状态变化。

## 练习

为健康接口增加应用版本字段，并补一个 API 测试。

## 面试检查点

解释为什么业务模块不直接读取全局环境变量，以及 AsyncSession 为什么不能跨并发任务共享。

