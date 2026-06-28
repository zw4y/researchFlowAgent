# 02 FastAPI 与 LLM

## 学习目标

掌握 HTTP 状态码、Pydantic 请求响应、OpenAI 兼容 API、Structured Output 和 Fake Provider。

## 代码地图

- `providers/base.py`：Provider Protocol。
- `providers/openai_compatible.py`：真实模型调用。
- `providers/fake.py`：可重复、零费用测试。
- `/api/v1/llm/structured`：结构化输出接口。

## 动手

分别提交 `paper_summary` 和 `research_plan`，观察同一个 prompt 如何被不同 schema 约束。

## 练习

增加 `experiment_summary` schema，并对必填字段做 Pydantic 校验。

## 面试检查点

说明 temperature、上下文窗口、JSON mode 与应用层校验的区别。

