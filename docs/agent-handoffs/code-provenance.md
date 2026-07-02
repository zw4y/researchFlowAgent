# Agent Code Provenance

This document separates the ResearchFlow evaluation changes by generating agent.
Git commit trailers remain the authoritative per-commit record.

## DeepSeek / ZCode

Remote branch:

`handoff/deepseek-v4/answer-evaluation`

Commits:

- `bcbbcb3` - answer-level evaluation framework
- `b174980` - initial handoff update
- `79ac8d9` - Qwen-turbo evaluation results
- `7a34815` - comprehensive handoff update
- `6ef8a79` - runtime and concurrency changes

Primary files:

- `backend/app/evaluation/answer_models.py`
- `backend/app/evaluation/answer_metrics.py`
- `backend/app/evaluation/answer_runner.py`
- `backend/app/evaluation/answer_cli.py`
- `backend/app/evaluation/verify_cases.py`
- `backend/tests/test_answer_evaluation.py`
- `docs/agent-handoffs/deepseek-v4-answer-evaluation.md`

The answer-level benchmark used Qwen-turbo at runtime, despite the DeepSeek agent
identity used for implementation.

## Codex / GPT-5

Remote branch:

`codex/evaluation-benchmark-integration`

Primary files:

- `backend/app/evaluation/__init__.py`
- `backend/app/evaluation/cli.py`
- `backend/app/evaluation/generate_cases.py`
- `backend/app/evaluation/metrics.py`
- `backend/app/evaluation/runner.py`
- `backend/tests/test_evaluation.py`
- `backend/tests/test_evaluation_generation.py`
- `backend/tests/test_evaluation_generation_resilience.py`
- `demo/evaluation/15-paper-cases.jsonl`
- `demo/evaluation/15-paper-test-cases.jsonl`
- `demo/evaluation/retrieval_cases.jsonl`
- `docs/evaluation.md`
- `docs/evaluation-baseline.md`
- `docs/evaluation-15-paper-test.md`
- `scripts/run-evaluation.ps1`

Codex also changed:

- `backend/app/services/retrieval.py` to expose retrieval traces and latency.
- `pyproject.toml` to register evaluation command-line entry points.
- `backend/app/evaluation/answer_runner.py` to fix type narrowing after the
  DeepSeek concurrency change.

## Shared-History Note

`backend/app/evaluation/models.py` was initially generated during the Codex
retrieval-evaluation work, but its first Git commit and the later
`independent_model_verified` modification were made by DeepSeek in `bcbbcb3`.
Git blame therefore attributes that file to the DeepSeek commit; this note records
the earlier generation history.

## Review Status

- DeepSeek commits: `Human-Reviewed: no`
- Codex commits: `Human-Reviewed: no`
- Full backend tests are required before merge.
- The answer benchmark labels still require user sampling before being described
  as human-verified ground truth.
