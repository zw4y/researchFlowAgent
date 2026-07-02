# 15-Paper Retrieval Evaluation

Measured on 2026-07-01 with the production retrieval profile:

- 15 indexed image-fusion papers.
- 120 generated evaluation cases.
- Deterministic hash split: 30 development and 90 held-out test cases.
- `text-embedding-v4` with 1024-dimensional vectors.
- Qdrant Top 20 retrieval and `qwen3-rerank` Top 6 selection.

## Held-Out Test Results

| Metric | Vector Top-6 | Reranked Top-6 | Delta |
| --- | ---: | ---: | ---: |
| Precision@6 | 14.81% | 17.41% | +2.59 pp |
| Recall@6 | 82.78% | 92.22% | +9.44 pp |
| MRR@6 | 0.6246 | 0.7511 | +0.1265 |
| Context-token savings vs full paper | 84.38% | 83.87% | -0.51 pp |

Additional results:

- Candidate Recall@20: **99.44%**
- Average full-paper context: **25,208 tokens**
- Average selected RAG context: **3,880 tokens**
- End-to-end retrieval and rerank latency: **608 ms P50**, **788 ms P95**
- Rerank provider success rate: **100%**
- Rerank Recall outcome: **15 wins, 4 losses, 71 ties**

## Confidence Intervals

Question-level bootstrap with 10,000 resamples:

| Delta | 95% confidence interval |
| --- | ---: |
| Precision@6 | +0.74 to +4.44 pp |
| Recall@6 | +1.11 to +17.78 pp |
| MRR@6 | +0.0413 to +0.2119 |

## Test Composition

The 90 test cases include:

- 34 quantitative-table questions.
- 20 factual questions.
- 9 architecture questions.
- 9 training questions.
- 9 ablation questions.
- 9 limitation questions.

Of these, 46 answers passed automatic numeric grounding against the cited PDF
pages. The remaining 44 semantic answers are marked `machine_generated` and
require human review before this benchmark can support answer-accuracy or
hallucination claims.

## Interpretation

These results support claims about retrieval quality, evidence ordering, latency,
and context-token reduction. They do not yet support claims about final-answer
accuracy, citation faithfulness, or superiority to a closed-book general model.
Those require human-verified reference answers and a separate generation
comparison.
