# 06 安全 SQL

## 学习目标

掌握 SQL CRUD、AST 解析、表字段白名单、只读事务和 CSV 数据导入。

## 代码地图

- `services/metrics.py`：CSV、SafeSQLValidator、只读执行。
- `demo/experiment_metrics.csv`：导入示例。
- `tests/test_safe_sql.py`：攻击与边界测试。

## 动手

导入指标后提问“不同实验的 accuracy 对比”；再尝试 DELETE、SELECT * 和多语句。

## 练习

允许 `GROUP BY metric_name`，同时保持函数与字段白名单。

## 面试检查点

说明仅靠 prompt 要求“不要写危险 SQL”为什么不构成安全边界。

