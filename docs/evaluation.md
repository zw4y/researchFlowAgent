# ResearchFlow Evaluation

ResearchFlow uses a versioned qrels dataset to compare the production retrieval
pipeline against simpler baselines. Do not claim quality improvements from a few
screenshots or hand-picked questions.

## Compared systems

1. **Vector Top-6**: the first six results returned by Qdrant.
2. **Vector Top-20 + Rerank Top-6**: the production DashScope rerank path.
3. **Full-paper context**: all indexed chunks from the selected papers, used only
   as the context-token baseline.

The vector and reranked variants use the same Top-20 candidate pool. This isolates
the effect of reranking. Full-paper context is not sent to the LLM during this
retrieval-only evaluation, so running the benchmark does not spend DeepSeek
generation tokens.

## Metrics

- **Candidate Recall@20**: whether the vector stage retrieves the annotated evidence.
- **Precision@6**: the share of six selected positions occupied by unique relevant
  pages or chunks.
- **Recall@6**: the share of annotated evidence represented in the final context.
- **MRR@6**: how early the first relevant result appears.
- **Context-token savings**: `1 - selected context tokens / full-paper tokens`.
- **Latency**: local vector retrieval time and external rerank time.
- **Rerank success rate**: distinguishes real reranking from provider fallback.

All averages are macro averages across questions. Precision and recall deltas are
absolute percentage-point changes, not relative percentages.

## Run

Stop the local API first because Qdrant Local Mode permits only one process to open
the same storage directory. Then run from the repository root:

```powershell
conda activate researchflow
python -m app.evaluation.cli --dataset demo/evaluation/retrieval_cases.jsonl
```

The command writes:

- `data/evaluations/latest/report.json` for machine processing.
- `data/evaluations/latest/report.md` for the README, portfolio, and interview review.

The committed demo dataset expects the two named demo papers to be uploaded and
indexed with the current profile. Add new JSONL cases only after manually verifying
their correct pages or chunk IDs.

## Interpretation

The report can support claims such as:

> On N annotated research questions, reranking improved Recall@6 by X percentage
> points and Precision@6 by Y percentage points, while RAG reduced paper-context
> tokens by Z% versus a full-paper prompt.

It does not by itself measure final-answer factual accuracy, generated-token cost,
or user satisfaction. Those require a separate answer-level dataset and either
human review or a calibrated judge model.
