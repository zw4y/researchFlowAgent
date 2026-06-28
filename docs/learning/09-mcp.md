# 09 MCP

## 学习目标

理解 MCP Tool、Resource、Prompt、Client、Server、stdio 与 Streamable HTTP。

## 代码地图

- `mcp_server.py`：FastMCP 与四个研究工具。
- `main.py`：MCP ASGI mount 和 session manager lifespan。
- `container.py`：MCP 与 REST 复用服务层。

## 动手

用 MCP Inspector 连接 `/mcp`，依次调用 `search_papers`、`ask_paper` 和 `get_citations`。

## 练习

增加 `paper://{paper_id}` Resource，返回论文元数据而不是执行动作。

## 面试检查点

区分 MCP Tools、Resources 和 Prompts 的控制方与适用场景。

