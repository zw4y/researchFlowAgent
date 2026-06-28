# 04 可溯源 RAG

## 学习目标

掌握 chunk、overlap、Embedding、向量检索、metadata filter、Top-K 和引用溯源。

## 代码地图

- `rag/pdf.py`：页面内 token chunk。
- `rag/vector_store.py`：内存与 Qdrant 实现。
- `services/retrieval.py`：查询向量与 Evidence。
- `agent/workflow.py`：Citation Check。

## 动手

修改 `RETRIEVAL_TOP_K` 和阈值，比较引用数量与噪声；确认 chunk 不跨页。

## 练习

实现一个可选 rerank 接口，输入候选 chunk，输出重新排序结果。

## 面试检查点

说明 overlap 太大、Top-K 太高和阈值太低分别会造成什么问题。

