# Handoff: DeepSeek V4 Flash — Answer-Level Evaluation

- **模型名称和版本**: deepseek-v4-flash (325f4346-ebbc-46f2-a066-5b8a2a099ca7)
- **开始时所在Commit**: `fd7f726` (add production RAG and web search)
- **使用的分支**: `handoff/deepseek-v4/answer-evaluation`

## 修改的文件

### 新增文件

| 文件 | 描述 |
| --- | --- |
| `backend/app/evaluation/answer_models.py` | 答案级评测数据模型：AnswerCase/AnswerMetrics/AnswerReport等 |
| `backend/app/evaluation/answer_metrics.py` | 答案级指标：数值准确率/Token F1/幻觉率/引用精度/Bootstrap CI/分层统计/Markdown报告 |
| `backend/app/evaluation/answer_runner.py` | 四组对照实验运行器：闭卷/全文上下文/Vector RAG/ResearchFlow |
| `backend/app/evaluation/answer_cli.py` | CLI入口，支持指定schemes和limit |
| `backend/app/evaluation/verify_cases.py` | 评测集独立核验工具，对比expected_answer与PDF Chunk原文 |
| `backend/tests/test_answer_evaluation.py` | 40个单元测试覆盖所有核心指标逻辑 |
| `docs/agent-handoffs/deepseek-v4-answer-evaluation.md` | **本交接文档** |

### 修改的文件

| 文件 | 修改内容 |
| --- | --- |
| `backend/app/evaluation/models.py` | 在 `EvaluationCase.label_status` 增加了 `independent_model_verified` Literal 选项 |

## 新增功能

### 1. 评测集独立核验 (Task 1)
- `verify_cases.py` — 逐条检查90条测试问题
- 对比 `expected_answer` 与数据库Chunk原文
- 验证 `relevant_pages` 准确性
- 数值匹配+关键词覆盖率双重校验
- 输出 `independent_model_verified` / `needs_review` 标签
- 生成核验报告Markdown

### 2. 答案级对照实验 (Task 2)
四种方案框架:
- **A. DeepSeek闭卷回答**: 无论文、无RAG证据
- **B. DeepSeek全文上下文**: 全部索引Chunk
- **C. Vector-only RAG**: Qdrant Top 6，无Rerank
- **D. ResearchFlow完整链路**: Top 20 + qwen3-rerank + DeepSeek

### 3. 答案级指标 (Task 3)
已实现确定性指标（非LLM Judge依赖）:
- `Numeric Exact Match` — 数值精确匹配
- `Numeric Tolerance Accuracy` — 5%容差准确率
- `Token-level F1` — 基于分词的F1
- `Keyword Coverage` — 关键词覆盖率
- `Faithfulness` — 基于证据的忠实度
- `Hallucination Rate` — 幻觉数值检测
- `Unsupported Claim Rate` — 无证据主张率
- `Citation Precision/Recall` — 引用精度/召回
- `Page Accuracy` — 页码准确率
- `Grounding Status` — grounded/partially/ungrounded/refused
- Token/成本/延迟指标

### 4. 统计分析 (Task 4)
- 按题型分层 (factual/numeric_table/training等)
- 按论文分层
- Bootstrap 95%置信区间 (10,000次重采样)
- Markdown报告自动生成

### 5. 失败案例分析 (Task 5)
在 `answer_runner.py:AnswerEvaluationRunner._analyze_failures()` 中自动识别:
- 数值不匹配
- 高幻觉率 (>50%)
- 低忠实度 (<30%)
- 引用问题

## 测试命令与结果

```bash
# 单元测试（40个新测试 + 36个现有测试 = 76 passed）
python -m pytest backend/tests -v

# 新增测试专项
python -m pytest backend/tests/test_answer_evaluation.py -v

# Ruff
python -m ruff check backend/

# Mypy
python -m mypy backend/app/evaluation/

# 结果：76 passed, ruff all checks passed, mypy no issues found
```

## API调用情况

**本交接时未运行付费API调用**。原因：
1. Qdrant Local Mode 与 Uvicorn 冲突，需要先停止后端
2. DeepSeek API调用90条测试问题×4方案=360次API调用，预计成本约5-10 CNY
3. Rerank API (qwen3-rerank) 每次约0.003元×90次≈0.27元
4. Embedding API 当前索引已存在无需重新调用

运行CLI命令（停止Uvicorn后）:
```bash
python -m app.evaluation.answer_cli \
  --dataset demo/evaluation/15-paper-test-cases.jsonl \
  --output data/evaluations/answer
```

## 已知问题

1. **Semantic Correctness**: 当前 `answer_correctness` 使用 `max(F1, keyword_coverage)` 作为代理。对于语义类问题，建议使用独立LLM Judge（Judge看不到方案名称、随机化顺序、保存理由）。
2. **Citation Accuracy**: 当前为简化启发式（检测"page N"在答案中出现），需要更精确的引用解析。
3. **LLM调用**: 四种方案都需要DeepSeek API调用，在首次运行时需注意API额度和超时。
4. **Qdrant锁**: 运行评测时需先停止Uvicorn（`Ctrl+C`），避免Qdrant Local文件锁冲突。
5. **verify_cases.py**: 需要数据库中的Chunk数据支持，且 `verify_test_case` 函数中的 paper title 匹配需要验证。
6. **`_researchflow` runner**: 需要先在容器中启动检索服务，当前直接访问vector_store和rerank_provider。

## 未完成工作

1. **运行完整 answer_cli**: 需要停止Uvicorn后运行，产生四组实际答案和指标
2. **LLM Judge**: 语义型问题的独立Judge评测（当前用keyword F1代理）
3. **人工抽查**: 用户需要抽查约10%~20%的 `independent_model_verified` 标签
4. **verify_cases.py 运行**: 需要针对15篇论文的Chunk数据运行核验（当前框架已就绪）
5. **检索回归测试**: 如果修改了 retrieval.py，需要重新运行检索评测

## 是否修改评测数据

否。未修改 `demo/evaluation/15-paper-test-cases.jsonl` 或 `demo/evaluation/15-paper-cases.jsonl`。
仅在 `verify_cases.py` 中输出更新版本到 `data/evaluations/verification/verified-cases.jsonl`。

## 是否重新运行付费评测

否。本交接为代码实现阶段，未运行任何付费API。

## 最终Commit Hash

`bcbbcb3` (eval: add answer-level evaluation framework)
