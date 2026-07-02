import pytest
from app.evaluation.metrics import (
    aggregate_results,
    evaluate_ranking,
    token_savings_ratio,
)
from app.evaluation.models import CaseResult, RankingMetrics


def test_evaluate_ranking_uses_unique_relevant_targets() -> None:
    relevant = {"paper-a:9", "paper-a:10"}
    retrieved = ["paper-a:9", "paper-a:9", "paper-a:4", "paper-a:10"]

    metrics = evaluate_ranking(retrieved, relevant, k=4)

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.reciprocal_rank == pytest.approx(1.0)
    assert metrics.hits == 2


def test_evaluate_ranking_reports_zero_without_hits() -> None:
    metrics = evaluate_ranking(["paper-a:1"], {"paper-a:9"}, k=6)

    assert metrics.precision == 0
    assert metrics.recall == 0
    assert metrics.reciprocal_rank == 0
    assert metrics.hits == 0


def test_token_savings_ratio_compares_rag_with_full_context() -> None:
    assert token_savings_ratio(selected_tokens=5_000, full_context_tokens=25_000) == pytest.approx(
        0.8
    )
    assert token_savings_ratio(selected_tokens=100, full_context_tokens=0) == 0


def test_aggregate_results_exposes_rerank_delta_and_token_savings() -> None:
    cases = [
        CaseResult(
            case_id="q1",
            query="question one",
            candidate_k=20,
            vector=RankingMetrics(precision=0.2, recall=0.5, reciprocal_rank=0.5, hits=1),
            reranked=RankingMetrics(
                precision=0.4,
                recall=1.0,
                reciprocal_rank=1.0,
                hits=2,
            ),
            candidate_recall=1.0,
            full_context_tokens=20_000,
            vector_context_tokens=4_000,
            reranked_context_tokens=3_000,
            vector_latency_ms=40,
            rerank_latency_ms=60,
            retrieval_status="reranked",
        ),
        CaseResult(
            case_id="q2",
            query="question two",
            candidate_k=20,
            vector=RankingMetrics(precision=0.1, recall=0.5, reciprocal_rank=1.0, hits=1),
            reranked=RankingMetrics(
                precision=0.2,
                recall=0.5,
                reciprocal_rank=1.0,
                hits=1,
            ),
            candidate_recall=0.5,
            full_context_tokens=30_000,
            vector_context_tokens=6_000,
            reranked_context_tokens=5_000,
            vector_latency_ms=50,
            rerank_latency_ms=70,
            retrieval_status="reranked",
        ),
    ]

    summary = aggregate_results(cases)

    assert summary.case_count == 2
    assert summary.vector_precision == pytest.approx(0.15)
    assert summary.reranked_precision == pytest.approx(0.3)
    assert summary.precision_delta == pytest.approx(0.15)
    assert summary.vector_recall == pytest.approx(0.5)
    assert summary.reranked_recall == pytest.approx(0.75)
    assert summary.recall_delta == pytest.approx(0.25)
    assert summary.reranked_token_savings == pytest.approx((0.85 + (5 / 6)) / 2)
