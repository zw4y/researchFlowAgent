# 08 联网研究

## 学习目标

理解外部 API 适配器、超时、降级、网页证据与私有论文证据的区分。

## 代码地图

- `providers/search.py`：Tavily adapter。
- `web_search_node`：只在路由命中且配置密钥时执行。
- Citation 的 `source_type`：paper 或 web。

## 动手

不设置密钥时运行论文问答；设置 Tavily 后提问近期信息，确认两种模式都可工作。

## 练习

加入 URL 域名白名单或黑名单，并在 Citation 中保存访问时间。

## 面试检查点

解释搜索摘要不是权威事实，以及为什么回答仍需展示原始 URL。

