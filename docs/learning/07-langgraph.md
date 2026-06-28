# 07 LangGraph

## 学习目标

掌握 State、Node、Edge、编译、运行配置与持久化 thread ID。

## 代码地图

- `agent/workflow.py`：AgentState 和六个节点。
- `providers/tool_selector.py`：路由前的工具计划。
- `agent_runs`、`messages`：数据库中的持久化运行历史。

## 动手

分别提出纯论文、论文加指标、论文加网页三个问题，比较 routes 与 tool_calls。

## 练习

增加 `human_review` 状态：当 grounding 为 unsupported 时暂停并请求人工确认。

## 面试检查点

说明显式工作流与 ReAct 自由循环的可控性、扩展性和成本差异。

