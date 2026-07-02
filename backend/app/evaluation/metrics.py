from statistics import fmean

from app.evaluation.models import CaseResult, EvaluationSummary, RankingMetrics


def evaluate_ranking(
    retrieved_keys: list[str],
    relevant_keys: set[str],
    *,
    k: int,
) -> RankingMetrics:
    ranked = retrieved_keys[:k]
    unique_hits = {key for key in ranked if key in relevant_keys}
    first_relevant_rank = next(
        (index for index, key in enumerate(ranked, start=1) if key in relevant_keys),
        None,
    )
    return RankingMetrics(
        precision=len(unique_hits) / k if k else 0,
        recall=len(unique_hits) / len(relevant_keys) if relevant_keys else 0,
        reciprocal_rank=1 / first_relevant_rank if first_relevant_rank else 0,
        hits=len(unique_hits),
    )


def token_savings_ratio(selected_tokens: int, full_context_tokens: int) -> float:
    if full_context_tokens <= 0:
        return 0
    return max(0.0, 1 - selected_tokens / full_context_tokens)


def aggregate_results(cases: list[CaseResult]) -> EvaluationSummary:
    if not cases:
        return EvaluationSummary(
            case_count=0,
            candidate_recall=0,
            vector_precision=0,
            reranked_precision=0,
            precision_delta=0,
            vector_recall=0,
            reranked_recall=0,
            recall_delta=0,
            vector_mrr=0,
            reranked_mrr=0,
            mrr_delta=0,
            vector_token_savings=0,
            reranked_token_savings=0,
            average_vector_latency_ms=0,
            average_rerank_latency_ms=0,
            rerank_success_rate=0,
        )

    vector_precision = fmean(item.vector.precision for item in cases)
    reranked_precision = fmean(item.reranked.precision for item in cases)
    vector_recall = fmean(item.vector.recall for item in cases)
    reranked_recall = fmean(item.reranked.recall for item in cases)
    vector_mrr = fmean(item.vector.reciprocal_rank for item in cases)
    reranked_mrr = fmean(item.reranked.reciprocal_rank for item in cases)
    return EvaluationSummary(
        case_count=len(cases),
        candidate_recall=fmean(item.candidate_recall for item in cases),
        vector_precision=vector_precision,
        reranked_precision=reranked_precision,
        precision_delta=reranked_precision - vector_precision,
        vector_recall=vector_recall,
        reranked_recall=reranked_recall,
        recall_delta=reranked_recall - vector_recall,
        vector_mrr=vector_mrr,
        reranked_mrr=reranked_mrr,
        mrr_delta=reranked_mrr - vector_mrr,
        vector_token_savings=fmean(
            token_savings_ratio(item.vector_context_tokens, item.full_context_tokens)
            for item in cases
        ),
        reranked_token_savings=fmean(
            token_savings_ratio(item.reranked_context_tokens, item.full_context_tokens)
            for item in cases
        ),
        average_vector_latency_ms=fmean(item.vector_latency_ms for item in cases),
        average_rerank_latency_ms=fmean(item.rerank_latency_ms for item in cases),
        rerank_success_rate=fmean(
            1.0 if item.retrieval_status == "reranked" else 0.0 for item in cases
        ),
    )
