# 05 Function Calling 与工具

## 学习目标

理解 JSON Schema、模型工具选择、参数校验、工具执行和失败降级。

## 代码地图

- `providers/tool_selector.py`：真实 OpenAI Function Calling。
- `tools/registry.py`：ToolDefinition、Pydantic 参数和执行轨迹。
- `/api/v1/tools`：查看提供给模型的工具 schema。

## 动手

查看工具 JSON Schema；提出同时需要论文和最新网页的问题，观察调用列表。

## 练习

增加 `compare_numbers` 工具，并限制最多比较 20 个值。

## 面试检查点

解释 Function Calling 不是“模型执行函数”，模型只产生结构化调用意图，执行权仍在应用。

