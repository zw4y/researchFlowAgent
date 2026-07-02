"""Tests for answer-level evaluation metrics, models, and verification."""


import pytest
from app.evaluation.answer_metrics import (
    aggregate_answer_metrics,
    check_citation_accuracy,
    check_grounding_status,
    compute_bootstrap_ci,
    detect_hallucinated_numbers,
    detect_unsupported_claims,
    evaluate_answer,
    extract_keywords,
    extract_numeric_values,
    format_markdown_report,
    keyword_coverage,
    normalize_answer,
    numeric_exact_match,
    numeric_tolerance_accuracy,
    stratify_by_paper,
    stratify_by_type,
    token_f1_score,
    tokenize,
)
from app.evaluation.answer_models import (
    AnswerCase,
    AnswerMetrics,
    AnswerReport,
    AnswerSchemeSummary,
)
from app.evaluation.verify_cases import (
    check_answer_in_source,
    extract_pages_from_answer,
)

# ─── Tokenization & Keyword Extraction ────────────────────────────────────────


def test_tokenize_english() -> None:
    tokens = tokenize("Attention layers improve accuracy by 5 percent.")
    assert "attention" in tokens
    assert "layers" in tokens
    assert "improve" in tokens
    assert "accuracy" in tokens


def test_tokenize_chinese() -> None:
    tokens = tokenize("注意力机制提升了准确率")
    assert "注" in tokens
    assert "意" in tokens
    assert "力" in tokens
    assert "机" in tokens
    assert "制" in tokens


def test_tokenize_mixed() -> None:
    tokens = tokenize("ISFM在MSRS数据集上达到Avg.Rank 1.14")
    assert "isfm" in tokens
    assert "msrs" in tokens
    assert "数" in tokens
    assert "据" in tokens


def test_extract_keywords_filters_stopwords() -> None:
    keywords = extract_keywords("the quick brown fox jumps over the lazy dog")
    assert "the" not in keywords
    assert "quick" in keywords
    assert "brown" in keywords


# ─── Numeric Extraction ───────────────────────────────────────────────────────


def test_extract_numeric_values_simple() -> None:
    values = extract_numeric_values("Accuracy: 91.5%, loss: 0.05")
    assert len(values) == 2
    # Check percentages
    pct_values = [v for v in values if v["is_percentage"]]
    assert len(pct_values) == 1
    assert abs(pct_values[0]["value"] - 91.5) < 0.01
    assert pct_values[0]["unit"] == "%"


def test_extract_numeric_values_with_units() -> None:
    values = extract_numeric_values("9.148M parameters, 608 ms latency")
    assert len(values) >= 2
    params = [v for v in values if abs(v["value"] - 9.148) < 0.001]
    assert len(params) >= 1


def test_extract_numeric_values_empty() -> None:
    assert extract_numeric_values("no numbers here") == []


# ─── Numeric Exact Match ──────────────────────────────────────────────────────


def test_numeric_exact_match_passes() -> None:
    matched, gen, ref = numeric_exact_match(
        "The accuracy is 91.5%",
        "Accuracy: 91.5%",
    )
    assert matched


def test_numeric_exact_match_fails() -> None:
    matched, gen, ref = numeric_exact_match(
        "The accuracy is 95%",
        "Accuracy: 91.5%",
    )
    assert not matched


def test_numeric_exact_match_no_numeric() -> None:
    matched, gen, ref = numeric_exact_match(
        "The model uses attention",
        "It uses transformer architecture",
    )
    assert matched  # No numeric values to match


# ─── Numeric Tolerance Accuracy ────────────────────────────────────────────────


def test_numeric_tolerance_exact() -> None:
    acc = numeric_tolerance_accuracy("value: 91.5%", "value: 91.5%")
    assert acc == pytest.approx(1.0)


def test_numeric_tolerance_within() -> None:
    acc = numeric_tolerance_accuracy("value: 91.5%", "value: 92.0%")
    assert acc == pytest.approx(1.0)  # Within 5%


def test_numeric_tolerance_outside() -> None:
    acc = numeric_tolerance_accuracy("value: 100%", "value: 50%")
    assert acc == pytest.approx(0.0)


# ─── Token F1 ─────────────────────────────────────────────────────────────────


def test_token_f1_identical() -> None:
    f1 = token_f1_score("The model achieves 91.5% accuracy", "The model achieves 91.5% accuracy")
    assert f1 == pytest.approx(1.0)


def test_token_f1_partial() -> None:
    f1 = token_f1_score("91.5% accuracy", "The model achieves 91.5% accuracy")
    assert 0 < f1 < 1.0


def test_token_f1_no_overlap() -> None:
    f1 = token_f1_score("completely different", "no match at all")
    # After stopword filtering, these truly have no keyword overlap
    assert f1 == pytest.approx(0.0)


# ─── Keyword Coverage ─────────────────────────────────────────────────────────


def test_keyword_coverage_full() -> None:
    cov = keyword_coverage(
        "ISFM introduces Interactive Spatial-Frequency Fusion Mamba",
        "ISFM Spatial-Frequency Fusion Mamba",
    )
    assert cov == pytest.approx(1.0)


def test_keyword_coverage_partial() -> None:
    cov = keyword_coverage(
        "The model uses attention",
        "ISFM spatial frequency fusion mamba architecture",
    )
    assert 0 <= cov <= 1.0


# ─── Citation Accuracy ────────────────────────────────────────────────────────


def test_citation_accuracy_with_citations() -> None:
    prec, rec, page_acc = check_citation_accuracy(
        "As shown in page 9 [P1], the result is 91.5%",
        [{"paper_title": "Paper A", "page": 9}],
    )
    assert prec > 0
    assert rec > 0


def test_citation_accuracy_no_citations() -> None:
    prec, rec, page_acc = check_citation_accuracy(
        "The result is 91.5%",
        [{"paper_title": "Paper A", "page": 9}],
    )
    assert prec == 0.0
    assert rec == 0.0


# ─── Grounding Status ─────────────────────────────────────────────────────────


def test_grounding_grounded() -> None:
    assert check_grounding_status("Based on [P1], the answer is...", 3) == "grounded"


def test_grounding_refused() -> None:
    assert check_grounding_status("I cannot answer this question", 0) == "refused"


def test_grounding_ungrounded_without_evidence() -> None:
    assert check_grounding_status("The answer is 42.", 0) == "ungrounded"


# ─── Hallucination Detection ──────────────────────────────────────────────────


def test_hallucination_detected() -> None:
    total, hall, rate = detect_hallucinated_numbers(
        "Accuracy is 99.9%",
        ["The reported accuracy is 91.5%"],
    )
    assert rate > 0


def test_hallucination_clean() -> None:
    total, hall, rate = detect_hallucinated_numbers(
        "Accuracy is 91.5%",
        ["The reported accuracy is 91.5% on MSRS"],
    )
    assert rate == pytest.approx(0.0)


# ─── Unsupported Claims ──────────────────────────────────────────────────────


def test_unsupported_claims_detected() -> None:
    claims, rate = detect_unsupported_claims(
        "The model achieves 99.9% accuracy. It uses a novel architecture.",
        ["The model trains for 100 epochs"],
    )
    assert rate > 0


def test_unsupported_claims_supported() -> None:
    claims, rate = detect_unsupported_claims(
        "The model uses spatial frequency fusion",
        ["spatial-frequency fusion mamba for multi-modal image fusion"],
    )
    assert rate < 0.5  # Some keywords should match


# ─── Normalize Answer ────────────────────────────────────────────────────────


def test_normalize_answer() -> None:
    norm = normalize_answer("  The   Answer: 91.5%!  ")
    assert "answer" in norm
    assert "91.5%" in norm
    assert "  " not in norm


# ─── Full Evaluation Pipeline ─────────────────────────────────────────────────


def test_evaluate_answer_numeric_case() -> None:
    metrics = evaluate_answer(
        case_id="test-01",
        scheme="researchflow",
        answer_type="numeric_table",
        generated_answer="The MSRS Avg.Rank is 1.14",
        expected_answer="Avg.Rank: 1.14",
        relevant_pages=[{"paper_title": "Paper A", "page": 9}],
        evidence_texts=["MSRS dataset Avg.Rank is 1.14"],
        input_tokens=100,
        output_tokens=20,
    )
    assert metrics.case_id == "test-01"
    assert metrics.scheme == "researchflow"
    assert metrics.numeric_exact_match  # 1.14 matches
    assert metrics.token_f1 > 0
    assert metrics.keyword_coverage > 0


def test_evaluate_answer_semantic_case() -> None:
    metrics = evaluate_answer(
        case_id="test-02",
        scheme="closed_book",
        answer_type="factual",
        generated_answer="The paper proposes a novel fusion method called ISFM",
        expected_answer="ISFM introduces Interactive Spatial-Frequency Fusion Mamba",
        relevant_pages=[{"paper_title": "Paper A", "page": 1}],
        evidence_texts=[],
    )
    assert metrics.case_id == "test-02"
    assert metrics.grounding_status == "ungrounded"  # No evidence, no citations


# ─── Aggregation ──────────────────────────────────────────────────────────────


def test_aggregate_answer_metrics_empty() -> None:

    summary = aggregate_answer_metrics([], "researchflow")
    assert summary.case_count == 0


def test_aggregate_answer_metrics_multiple() -> None:
    metrics = [
        AnswerMetrics(
            case_id="c1",
            scheme="researchflow",
            answer_type="factual",
            numeric_exact_match=True,
            answer_correctness=0.9,
            token_f1=0.8,
            faithfulness=0.9,
            hallucination_rate=0.1,
            unsupported_claim_rate=0.1,
            citation_precision=0.5,
            citation_recall=0.5,
            page_accuracy=0.5,
            correct_refusal=False,
            should_refuse=False,
            grounding_status="grounded",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
            api_cost_cny=0.001,
            token_savings_ratio=0.8,
        ),
        AnswerMetrics(
            case_id="c2",
            scheme="researchflow",
            answer_type="numeric_table",
            numeric_exact_match=False,
            answer_correctness=0.5,
            token_f1=0.4,
            faithfulness=0.5,
            hallucination_rate=0.5,
            unsupported_claim_rate=0.5,
            citation_precision=0.0,
            citation_recall=0.0,
            page_accuracy=0.0,
            correct_refusal=False,
            should_refuse=False,
            grounding_status="partially_grounded",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
            api_cost_cny=0.002,
            token_savings_ratio=0.5,
        ),
    ]

    summary = aggregate_answer_metrics(metrics, "researchflow")
    assert summary.case_count == 2
    assert summary.answer_correctness == pytest.approx(0.7)
    assert summary.token_f1 == pytest.approx(0.6)
    assert summary.avg_input_tokens == pytest.approx(150)
    assert summary.avg_output_tokens == pytest.approx(75)


# ─── Bootstrap CI ────────────────────────────────────────────────────────────


def test_bootstrap_ci_returns_interval() -> None:
    a = [
        AnswerMetrics(case_id=f"c{i}", scheme="vector_rag", answer_type="factual",
                       answer_correctness=0.5, token_f1=0.5)
        for i in range(10)
    ]
    b = [
        AnswerMetrics(case_id=f"c{i}", scheme="researchflow", answer_type="factual",
                       answer_correctness=0.8, token_f1=0.8)
        for i in range(10)
    ]
    ci = compute_bootstrap_ci(a, b, "answer_correctness", n_resamples=100)
    assert len(ci) == 2
    assert ci[0] <= ci[1]  # Lower <= Upper


# ─── Stratification ──────────────────────────────────────────────────────────


def make_test_case(case_id: str, answer_type: str, scheme_metrics: tuple) -> AnswerCase:
    scheme, correct, f1 = scheme_metrics
    case = AnswerCase(
        case_id=case_id,
        query="test",
        paper_titles=["Paper A"],
        answer_type=answer_type,  # type: ignore[arg-type]
    )
    case.metrics[scheme] = AnswerMetrics(
        case_id=case_id,
        scheme=scheme,
        answer_type=answer_type,  # type: ignore[arg-type]
        answer_correctness=correct,
        token_f1=f1,
    )
    return case


def test_stratify_by_type() -> None:
    cases = [
        make_test_case("c1", "factual", ("researchflow", 0.9, 0.8)),
        make_test_case("c2", "numeric_table", ("researchflow", 0.8, 0.7)),
        make_test_case("c3", "factual", ("researchflow", 0.7, 0.6)),
    ]
    stratified = stratify_by_type(cases, "researchflow")
    assert "factual" in stratified
    assert stratified["factual"].case_count == 2
    assert stratified["factual"].answer_correctness == pytest.approx(0.8)


def test_stratify_by_paper() -> None:
    cases = [
        make_test_case("c1", "factual", ("researchflow", 0.9, 0.8)),
    ]
    stratified = stratify_by_paper(cases, "researchflow")
    assert "Paper A" in stratified


# ─── Verification Tools ──────────────────────────────────────────────────────


def test_extract_pages_from_answer() -> None:
    pages = extract_pages_from_answer("On page 9, the result shows... and page 12")
    assert 9 in pages
    assert 12 in pages


def test_check_answer_in_source_keywords_match() -> None:
    ok, findings = check_answer_in_source(
        "ISFM uses spatial frequency fusion",
        {
            1: (
                "ISFM introduces Interactive Spatial-Frequency Fusion Mamba "
                "for multi-modal image fusion"
            ),
        },
        [1],
    )
    assert ok


def test_check_answer_in_source_numeric_match() -> None:
    ok, findings = check_answer_in_source(
        "The accuracy is 91.5%",
        {9: "On MSRS dataset, the accuracy reaches 91.5%"},
        [9],
    )
    assert ok


def test_check_answer_in_source_no_match() -> None:
    ok, findings = check_answer_in_source(
        "Accuracy is 99.9%",
        {1: "The paper discusses image fusion techniques"},
        [1],
    )
    assert not ok


# ─── Format Markdown Report ──────────────────────────────────────────────────


def test_format_markdown_report_basic() -> None:
    report = AnswerReport(
        dataset_path="test.jsonl",
        schemes=["researchflow", "vector_rag"],
        case_count=1,
    )
    report.summaries["researchflow"] = AnswerSchemeSummary(
        scheme="researchflow", case_count=1,
        answer_correctness=0.85, token_f1=0.75,
    )
    report.summaries["vector_rag"] = AnswerSchemeSummary(
        scheme="vector_rag", case_count=1,
        answer_correctness=0.65, token_f1=0.55,
    )
    md = format_markdown_report(report)
    assert "Answer-Level Evaluation" in md
    assert "researchflow" in md.lower() or "ResearchFlow" in md
    assert "85.00%" in md or "85.00" in md
