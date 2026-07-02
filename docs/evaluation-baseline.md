# Evaluation Baseline: 2026-07-01

This baseline was measured with:

- 10 manually page-labeled questions over two indexed papers.
- `text-embedding-v4`, 1024 dimensions.
- Qdrant vector retrieval with Top 20 candidates.
- `qwen3-rerank` selecting Top 6 passages.
- Index profile `b2e91fe6e7377f64`.

## Results

| Metric | Vector Top-6 | Reranked Top-6 | Delta |
| --- | ---: | ---: | ---: |
| Precision@6 | 15.00% | 11.67% | -3.33 pp |
| Recall@6 | 90.00% | 70.00% | -20.00 pp |
| MRR@6 | 0.6250 | 0.7000 | +0.0750 |
| Context-token savings vs full paper | 84.20% | 83.51% | -0.69 pp |

Additional measurements:

- Candidate Recall@20: **100.00%**
- Rerank API success rate: **100.00%**
- Average vector retrieval latency: **260.5 ms**
- Average rerank latency: **343.1 ms**

## Interpretation

The embedding stage found every annotated evidence page in its Top 20 candidate
pool. Restricting the prompt to six passages reduced paper-context input by about
84% compared with sending every indexed chunk.

The reranker improved first-hit ordering, but reduced final Recall@6 on this small
dataset. It dropped evidence for training-configuration and mixed
training-efficiency questions. This is a measured regression, not a positive result
to hide. Before claiming a rerank improvement, ResearchFlow needs:

1. Separate development and held-out test qrels.
2. More questions covering tables, architecture, training, and cross-paper queries.
3. Rerank instruction or rank-fusion experiments tuned only on the development set.
4. A new frozen test report showing that the regression is removed.

This baseline demonstrates why the evaluation gate is part of the product: a
successful provider response does not imply better retrieval quality.
