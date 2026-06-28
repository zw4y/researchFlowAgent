# 03 论文摄取

## 学习目标

理解文件上传、哈希去重、后台任务、PDF 页解析和任务状态机。

## 代码地图

- `services/papers.py`：大小、扩展名、PDF magic bytes 与 SHA-256。
- `services/ingestion.py`：queued、processing、completed、failed。
- `rag/pdf.py`：逐页文本提取。

## 动手

连续上传同一文件两次；再上传扫描版 PDF，观察重复标识和 OCR 错误。

## 练习

为失败任务增加“重新处理”接口，确保不会重复创建论文记录。

## 面试检查点

解释为什么 BackgroundTasks 不是生产级任务队列，以及何时迁移到 Celery/Arq。

