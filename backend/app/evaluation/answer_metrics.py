"""Answer-level evaluation metrics: deterministic, rule-based, and LLM Judge.

Priority is given to deterministic rules for numeric questions.
LLM Judge is used only for semantic correctness when no rule applies.
"""

import re
from collections import Counter
from collections.abc import Callable
from statistics import fmean
from typing import Any, Literal

from app.evaluation.answer_models import (
    AnswerCase,
    AnswerMetrics,
    AnswerReport,
    AnswerScheme,
    AnswerSchemeSummary,
    AnswerType,
    GroundingStatus,
)

# ─── Deterministic Helpers ────────────────────────────────────────────────────

# Chinese and English stopwords for keyword extraction
_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "although", "though",
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "哪", "为什么", "因为", "所以", "但是", "然而", "虽然",
    "如果", "而且", "或者", "那么", "然后", "又", "还", "已经", "可以",
    "这个", "那个", "这些", "那些", "每个", "所有", "一些", "其他",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text into words (Chinese characters + English words)."""
    tokens: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fff]|[a-z\d]+", text.lower()):
        tokens.append(match.group(0))
    return tokens


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text."""
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1}


def extract_numeric_values(text: str) -> list[dict[str, Any]]:
    """Extract numeric values with units from text."""
    values: list[dict[str, Any]] = []
    # Match numbers with optional percentage, units
    for match in re.finditer(
        r"(?<![A-Za-z0-9])[-+]?\d+(?:[.,]\d+)?\s*(%|M|K|B|GB|MB|KB|"
        r"ms|s|h|mm|cm|m|px|dpi|fps)?",
        text,
    ):
        raw = match.group(0).strip()
        cleaned = raw.replace(",", "").replace(" ", "")
        # Extract unit from regex group 1 (optional)
        unit_suffix = (match.group(1) or "").strip()
        is_percentage = unit_suffix == "%"
        unit = unit_suffix or ""
        numeric_str = cleaned
        if unit_suffix:
            numeric_str = cleaned[: -len(unit_suffix)].strip()
        try:
            value = float(numeric_str)
            values.append(
                {"raw": raw, "value": value, "unit": unit, "is_percentage": is_percentage}
            )
        except ValueError:
            continue
    return values


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff%.,]", "", text)
    return text.strip()


# ─── Individual Metrics ───────────────────────────────────────────────────────


def numeric_exact_match(
    generated: str,
    reference: str,
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """Check if all numeric values in reference appear in generated answer."""
    gen_values = extract_numeric_values(generated)
    ref_values = extract_numeric_values(reference)

    if not ref_values:
        return True, gen_values, ref_values  # No numeric values to match

    matched = all(
        any(
            abs(gen["value"] - ref["value"]) < 0.01
            and gen["is_percentage"] == ref["is_percentage"]
            for gen in gen_values
        )
        for ref in ref_values
    )
    return matched, gen_values, ref_values


def numeric_tolerance_accuracy(
    generated: str,
    reference: str,
    tolerance: float = 0.05,
) -> float:
    """What fraction of reference numeric values are matched within tolerance."""
    gen_values = extract_numeric_values(generated)
    ref_values = extract_numeric_values(reference)

    if not ref_values:
        return 1.0

    matched = 0
    for ref in ref_values:
        for gen in gen_values:
            if gen["is_percentage"] == ref["is_percentage"]:
                if ref["value"] == 0:
                    if abs(gen["value"]) < tolerance:
                        matched += 1
                        break
                elif abs(gen["value"] - ref["value"]) / abs(ref["value"]) <= tolerance:
                    matched += 1
                    break
    return matched / len(ref_values)


def token_f1_score(generated: str, reference: str) -> float:
    """Compute token-level F1 between generated and reference answer."""
    gen_tokens = Counter(tokenize(generated))
    ref_tokens = Counter(tokenize(reference))

    intersection = sum((gen_tokens & ref_tokens).values())
    gen_sum = sum(gen_tokens.values())
    ref_sum = sum(ref_tokens.values())

    if gen_sum == 0 or ref_sum == 0:
        return 0.0

    precision = intersection / gen_sum
    recall = intersection / ref_sum
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(generated: str, reference: str) -> float:
    """What fraction of reference keywords appear in the generated answer."""
    gen_lower = generated.lower()
    ref_keywords = extract_keywords(reference)

    if not ref_keywords:
        return 1.0

    covered = sum(1 for kw in ref_keywords if kw in gen_lower)
    return covered / len(ref_keywords)


def check_citation_accuracy(
    answer_text: str,
    relevant_pages: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """Check citation precision, recall, and page accuracy.

    Citation format: [P1], [P2], etc. referencing evidence.
    Returns (precision, recall, page_accuracy).
    """
    cited_markers = re.findall(r"\[P(\d+)]", answer_text)
    has_citations = len(cited_markers) > 0

    if not has_citations or not relevant_pages:
        return 0.0, 0.0, 0.0

    # Expected pages from the test case
    expected_pages = {p["page"] for p in relevant_pages}

    # Simple heuristic: count how many cited pages match expected pages
    # For proper citation precision, we'd need to parse structured citations
    # This is a simplified check
    citation_precision = 0.0
    citation_recall = 0.0
    page_accuracy = 0.0

    if cited_markers:
        # Approximate: check if the answer mentions the right page numbers
        pages_mentioned = set()
        for page_match in re.finditer(r"page\s*(\d+)", answer_text.lower()):
            pages_mentioned.add(int(page_match.group(1)))

        if pages_mentioned:
            correct = len(pages_mentioned & expected_pages)
            citation_precision = correct / len(pages_mentioned) if pages_mentioned else 0.0
            citation_recall = correct / len(expected_pages) if expected_pages else 0.0
            page_accuracy = citation_precision  # same as precision for page-level

    return citation_precision, citation_recall, page_accuracy


def check_grounding_status(answer_text: str, evidence_count: int) -> GroundingStatus:
    """Determine grounding status from answer text and evidence."""
    if not answer_text or answer_text.startswith("I cannot answer"):
        return "refused"

    has_citations = bool(re.search(r"\[P\d+]", answer_text))
    has_evidence = evidence_count > 0

    if has_citations and has_evidence:
        return "grounded"
    elif has_evidence:
        return "partially_grounded"
    elif not has_evidence and has_citations:
        return "ungrounded"
    return "ungrounded"


def detect_unsupported_claims(
    answer_text: str,
    evidence_texts: list[str],
) -> tuple[list[str], float]:
    """Detect claims in answer not supported by evidence.

    Returns (unsupported_claims, unsupported_rate).
    """
    if not evidence_texts:
        # All claims are unsupported if no evidence provided
        sentences = re.split(r"[.。!！?？\n]", answer_text)
        filtered = [s.strip() for s in sentences if len(s.strip()) > 10]
        return filtered, 1.0 if filtered else 0.0

    combined_evidence = " ".join(evidence_texts).lower()
    sentences = re.split(r"[.。!！?？\n]", answer_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    unsupported: list[str] = []
    for sentence in sentences:
        # Check if key nouns/numbers from sentence appear in evidence
        keywords = extract_keywords(sentence)
        # For numeric sentences, check if any numbers appear in evidence
        nums = extract_numeric_values(sentence)
        supported = False
        if keywords:
            kw_matches = sum(1 for kw in keywords if kw in combined_evidence)
            if kw_matches / len(keywords) > 0.3:  # 30% keyword overlap
                supported = True
        if nums and not supported:
            # Check if any numeric value appears in evidence
            for num in nums:
                if str(num["value"]) in combined_evidence:
                    supported = True
                    break
        if not supported:
            unsupported.append(sentence)

    rate = len(unsupported) / len(sentences) if sentences else 0.0
    return unsupported, rate


def detect_hallucinated_numbers(
    answer_text: str,
    evidence_texts: list[str],
) -> tuple[int, int, float]:
    """Count hallucinated numeric values in answer vs evidence.

    Returns (total_numbers, hallucinated_numbers, hallucination_rate).
    """
    answer_nums = extract_numeric_values(answer_text)
    if not answer_nums:
        return 0, 0, 0.0

    combined_evidence = " ".join(evidence_texts)
    evidence_nums = {n["value"] for n in extract_numeric_values(combined_evidence)}

    hallucinated = sum(
        1 for num in answer_nums if num["value"] not in evidence_nums
    )
    return len(answer_nums), hallucinated, hallucinated / len(answer_nums) if answer_nums else 0.0


# ─── Main Evaluation Function ─────────────────────────────────────────────────


def evaluate_answer(
    *,
    case_id: str,
    scheme: AnswerScheme,
    answer_type: AnswerType,
    generated_answer: str,
    expected_answer: str,
    relevant_pages: list[dict[str, Any]],
    evidence_texts: list[str],
    input_tokens: int = 0,
    output_tokens: int = 0,
    full_context_tokens: int = 0,
    rag_context_tokens: int = 0,
    api_cost_cny: float = 0.0,
    latency_ms: int = 0,
    token_savings_ratio: float = 0.0,
    judge_method: Literal["deterministic", "llm_judge", "rule_based"] = "rule_based",
    judge_reason: str = "",
) -> AnswerMetrics:
    """Compute all answer-level metrics for a single case."""
    gen_norm = normalize_answer(generated_answer)
    ref_norm = normalize_answer(expected_answer)

    # Numeric metrics
    num_exact, gen_nums, ref_nums = numeric_exact_match(gen_norm, ref_norm)
    num_tolerance = numeric_tolerance_accuracy(gen_norm, ref_norm)

    # Token/keyword metrics
    f1 = token_f1_score(gen_norm, ref_norm)
    kw_cov = keyword_coverage(gen_norm, ref_norm)

    # Citation metrics
    cit_prec, cit_rec, page_acc = check_citation_accuracy(
        generated_answer, relevant_pages
    )

    # Grounding
    grounding = check_grounding_status(generated_answer, len(evidence_texts))

    # Hallucination and unsupported claims
    _, hall_count, hall_rate = detect_hallucinated_numbers(
        generated_answer, evidence_texts
    )
    unsupported_claims, unsupported_rate = detect_unsupported_claims(
        generated_answer, evidence_texts
    )

    # Refusal detection
    should_refuse = expected_answer == "" or expected_answer is None
    refused = grounding == "refused"
    correct_refusal = should_refuse and refused

    # Answer correctness: use F1 as proxy for semantic correctness
    # (LLM Judge would replace this for semantic questions)
    answer_correctness = f1 if answer_type == "numeric_table" else max(f1, kw_cov)

    return AnswerMetrics(
        case_id=case_id,
        scheme=scheme,
        answer_type=answer_type,
        numeric_exact_match=num_exact,
        numeric_tolerance_accuracy=num_tolerance,
        numeric_values_in_answer=len(gen_nums),
        numeric_values_in_reference=len(ref_nums),
        answer_correctness=answer_correctness,
        token_f1=f1,
        keyword_coverage=kw_cov,
        faithfulness=1.0 - unsupported_rate,
        hallucination_rate=hall_rate,
        unsupported_claim_rate=unsupported_rate,
        citation_precision=cit_prec,
        citation_recall=cit_rec,
        page_accuracy=page_acc,
        correct_refusal=correct_refusal,
        should_refuse=should_refuse,
        grounding_status=grounding,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        full_context_tokens=full_context_tokens,
        rag_context_tokens=rag_context_tokens,
        api_cost_cny=api_cost_cny,
        latency_ms=latency_ms,
        token_savings_ratio=token_savings_ratio,
        judge_method=judge_method,
        judge_reason=judge_reason,
    )


# ─── Aggregation ──────────────────────────────────────────────────────────────


def aggregate_answer_metrics(
    all_metrics: list[AnswerMetrics],
    scheme: AnswerScheme,
) -> AnswerSchemeSummary:
    """Aggregate metrics across all cases for one scheme."""
    if not all_metrics:
        return AnswerSchemeSummary(scheme=scheme, case_count=0)

    case_count = len(all_metrics)
    numeric_cases = [m for m in all_metrics if m.numeric_values_in_reference > 0]

    return AnswerSchemeSummary(
        scheme=scheme,
        case_count=case_count,
        numeric_exact_match=(
            fmean(m.numeric_exact_match for m in numeric_cases)
            if numeric_cases else 0.0
        ),
        numeric_tolerance_accuracy=(
            fmean(m.numeric_tolerance_accuracy for m in numeric_cases)
            if numeric_cases else 0.0
        ),
        answer_correctness=fmean(m.answer_correctness for m in all_metrics),
        token_f1=fmean(m.token_f1 for m in all_metrics),
        faithfulness=fmean(m.faithfulness for m in all_metrics),
        hallucination_rate=fmean(m.hallucination_rate for m in all_metrics),
        unsupported_claim_rate=fmean(m.unsupported_claim_rate for m in all_metrics),
        citation_precision=fmean(m.citation_precision for m in all_metrics),
        citation_recall=fmean(m.citation_recall for m in all_metrics),
        page_accuracy=fmean(m.page_accuracy for m in all_metrics),
        correct_refusal_rate=fmean(m.correct_refusal for m in all_metrics),
        grounding_rate=(
            sum(1 for m in all_metrics if m.grounding_status == "grounded")
            / case_count
            if case_count
            else 0.0
        ),
        avg_input_tokens=fmean(m.input_tokens for m in all_metrics),
        avg_output_tokens=fmean(m.output_tokens for m in all_metrics),
        avg_latency_ms=fmean(m.latency_ms for m in all_metrics),
        avg_api_cost_cny=fmean(m.api_cost_cny for m in all_metrics),
        avg_token_savings=fmean(m.token_savings_ratio for m in all_metrics),
    )


def compute_bootstrap_ci(
    metrics_a: list[AnswerMetrics],
    metrics_b: list[AnswerMetrics],
    metric_key: str,
    n_resamples: int = 10_000,
    ci: float = 0.95,
) -> list[float]:
    """Compute bootstrap confidence interval for the delta of a metric."""
    import random

    values_a = [getattr(m, metric_key) for m in metrics_a]
    values_b = [getattr(m, metric_key) for m in metrics_b]

    n = len(values_a)
    if n == 0:
        return [0.0, 0.0]

    deltas: list[float] = []
    for _ in range(n_resamples):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        sample_a = [values_a[i] for i in indices]
        sample_b = [values_b[i] for i in indices]
        deltas.append(fmean(sample_b) - fmean(sample_a))

    deltas.sort()
    lower_index = int(n_resamples * (1 - ci) / 2)
    upper_index = int(n_resamples * (1 + ci) / 2)
    return [deltas[lower_index], deltas[upper_index]]


def stratify_by_type(
    cases: list[AnswerCase],
    scheme: AnswerScheme,
) -> dict[str, AnswerSchemeSummary]:
    """Stratify metrics by answer type."""
    by_type: dict[str, list[AnswerMetrics]] = {}
    for case in cases:
        atype = case.answer_type
        if atype not in by_type:
            by_type[atype] = []
        if scheme in case.metrics:
            by_type[atype].append(case.metrics[scheme])

    return {
        atype: aggregate_answer_metrics(metrics, scheme)
        for atype, metrics in by_type.items()
        if metrics
    }


def stratify_by_paper(
    cases: list[AnswerCase],
    scheme: AnswerScheme,
) -> dict[str, AnswerSchemeSummary]:
    """Stratify metrics by paper title."""
    by_paper: dict[str, list[AnswerMetrics]] = {}
    for case in cases:
        for title in case.paper_titles:
            if title not in by_paper:
                by_paper[title] = []
            if scheme in case.metrics:
                by_paper[title].append(case.metrics[scheme])

    return {
        title: aggregate_answer_metrics(metrics, scheme)
        for title, metrics in by_paper.items()
        if metrics
    }


def format_percent(value: float) -> str:
    """Format a ratio as percentage."""
    return f"{value * 100:.2f}%"


def format_markdown_report(report: AnswerReport) -> str:
    """Generate a markdown report from the answer evaluation."""
    lines = [
        "# ResearchFlow Answer-Level Evaluation",
        "",
        f"- Generated: `{report.generated_at.isoformat()}`",
        f"- Dataset: `{report.dataset_path}`",
        f"- Cases: `{report.case_count}`",
        "",
        "## Overall Summary",
        "",
        "| Metric | " + " | ".join(report.schemes) + " |",
        "| --- | " + " | ".join("---:" for _ in report.schemes) + " |",
    ]

    metrics_rows = [
        ("Numeric Exact Match", "numeric_exact_match"),
        ("Numeric Tolerance Acc.", "numeric_tolerance_accuracy"),
        ("Answer Correctness", "answer_correctness"),
        ("Token F1", "token_f1"),
        ("Faithfulness", "faithfulness"),
        ("Hallucination Rate", "hallucination_rate"),
        ("Unsupported Claim Rate", "unsupported_claim_rate"),
        ("Citation Precision", "citation_precision"),
        ("Citation Recall", "citation_recall"),
        ("Page Accuracy", "page_accuracy"),
        ("Grounding Rate", "grounding_rate"),
    ]

    for label, key in metrics_rows:
        row = [label]
        for scheme in report.schemes:
            summary = report.summaries.get(scheme)
            if summary:
                row.append(format_percent(getattr(summary, key)))
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("| Metric | " + " | ".join(report.schemes) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in report.schemes) + " |")

    cost_rows: list[tuple[str, str, str | Callable[[float], str]]] = [
        ("Avg Input Tokens", "avg_input_tokens", "{:.0f}"),
        ("Avg Output Tokens", "avg_output_tokens", "{:.0f}"),
        ("Avg Latency (ms)", "avg_latency_ms", "{:.0f}"),
        ("Avg API Cost (CNY)", "avg_api_cost_cny", "{:.4f}"),
        ("Avg Token Savings", "avg_token_savings", format_percent),
    ]

    for label, key, fmt in cost_rows:
        row = [label]
        for scheme in report.schemes:
            summary = report.summaries.get(scheme)
            if summary:
                val = getattr(summary, key)
                row.append(fmt(val) if callable(fmt) else str(fmt).format(val))
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # Bootstrap confidence intervals
    if report.bootstrap_ci:
        lines.extend([
            "",
            "## Bootstrap 95% Confidence Intervals",
            "",
            "ResearchFlow vs Vector RAG deltas:",
            "",
            "| Metric | Lower | Upper |",
            "| --- | ---: | ---: |",
        ])
        for metric_key, interval in report.bootstrap_ci.items():
            if len(interval) == 2:
                lines.append(
                    f"| {metric_key} | {interval[0]*100:.2f}% | {interval[1]*100:.2f}% |"
                )

    # Stratified by answer type
    if report.by_answer_type:
        lines.extend([
            "",
            "## Results by Answer Type",
            "",
        ])
        for atype, scheme_dict in report.by_answer_type.items():
            lines.append(f"### {atype}")
            lines.append("")
            lines.append("| Metric | " + " | ".join(report.schemes) + " |")
            lines.append("| --- | " + " | ".join("---:" for _ in report.schemes) + " |")
            for label, key in metrics_rows:
                row = [label]
                for scheme in report.schemes:
                    summary = scheme_dict.get(scheme)
                    if summary:
                        row.append(format_percent(getattr(summary, key)))
                    else:
                        row.append("N/A")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # Failure analysis
    if report.failures:
        lines.extend([
            "## Failure Analysis",
            "",
        ])
        for failure in report.failures:
            lines.append(f"- **{failure.get('case_id', 'unknown')}** "
                         f"({failure.get('scheme', 'N/A')}): "
                         f"{failure.get('description', 'N/A')}")

    lines.append("")
    return "\n".join(lines)
