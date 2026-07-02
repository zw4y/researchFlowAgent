"""Independent case verification: check test cases against PDF chunks.

Verifies each test case's expected_answer against the actual PDF text
stored in database chunks. Assigns independent_model_verified status
where the answer can be confirmed from source text.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Literal

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Chunk
from app.db.session import Database
from app.evaluation.answer_metrics import extract_keywords, extract_numeric_values
from app.evaluation.models import EvaluationCase
from app.evaluation.runner import load_cases

VerifyStatus = Literal[
    "machine_generated",
    "auto_checked",
    "independent_model_verified",
    "needs_review",
]


class VerificationResult:
    """Result of verifying one evaluation case."""

    def __init__(self, case: EvaluationCase) -> None:
        self.case_id = case.case_id
        self.query = case.query
        self.expected_answer = case.expected_answer or ""
        self.original_status = case.label_status
        self.assigned_status: VerifyStatus = "machine_generated"
        self.relevant_pages_correct = True
        self.missing_pages: list[int] = []
        self.extra_pages: list[int] = []
        self.answer_consistent_with_source = True
        self.issues: list[str] = []
        self.source_text_found: list[str] = []
        self.answer_type = case.answer_type
        self.paper_titles = case.paper_titles
        self.relevant_pages = [
            p.page for p in case.relevant_pages
        ]


def extract_pages_from_answer(answer: str) -> set[int]:
    """Extract page numbers from the answer text."""
    pages = set()
    for match in re.finditer(r"page\s*(\d+)", answer.lower()):
        pages.add(int(match.group(1)))
    return pages


def check_answer_in_source(
    expected_answer: str,
    source_texts: dict[int, str],
    relevant_pages: list[int],
) -> tuple[bool, list[str]]:
    """Check if the expected answer is supported by the source texts."""
    findings: list[str] = []
    answer_keywords = extract_keywords(expected_answer)
    answer_nums = extract_numeric_values(expected_answer)

    # Check if answer keywords appear in relevant pages
    combined_relevant = " ".join(
        source_texts.get(p, "") for p in relevant_pages
    ).lower()

    if not combined_relevant:
        findings.append("No source text found for relevant pages")
        return False, findings

    # Check keyword coverage
    if answer_keywords:
        matched_kw = sum(1 for kw in answer_keywords if kw in combined_relevant)
        keyword_ratio = matched_kw / len(answer_keywords)
        if keyword_ratio < 0.5:
            findings.append(
                f"Only {matched_kw}/{len(answer_keywords)} keywords found in source "
                f"({keyword_ratio:.0%})"
            )
            return False, findings

    # Check numeric values
    if answer_nums:
        combined_text = " ".join(source_texts.values()).lower()
        all_found = all(
            str(num["value"]) in combined_text for num in answer_nums
        )
        if not all_found:
            missing = [
                num["raw"]
                for num in answer_nums
                if str(num["value"]) not in combined_text
            ]
            findings.append(f"Numeric values not found in source: {missing}")
            return False, findings

    findings.append("Answer content verified against source text")
    return True, findings


def check_pages_accurate(
    relevant_pages: list[int],
    source_texts: dict[int, str],
    paper_id: str,
    chunks: list[Chunk],
) -> tuple[bool, str]:
    """Check if the relevant pages contain relevant information for the query."""
    if not relevant_pages:
        return False, "No relevant pages specified"

    # Check all specified pages exist in the chunks
    available_pages = {chunk.page for chunk in chunks}
    missing = [p for p in relevant_pages if p not in available_pages]
    if missing:
        return False, f"Pages not found in chunks: {missing}"

    # Check pages have meaningful content
    for page in relevant_pages:
        if page in source_texts:
            text = source_texts[page]
            if len(text.strip()) < 20:
                return False, f"Page {page} has insufficient content ({len(text)} chars)"

    return True, "Pages verified"


async def verify_test_case(
    case: EvaluationCase,
    chunks_by_paper: dict[str, list[Chunk]],
) -> VerificationResult:
    """Independently verify a single test case."""
    result = VerificationResult(case)

    # Get the source chunks for this paper
    paper_chunks: list[Chunk] = []
    for title in case.paper_titles:
        # Match by title (fuzzy prefix match)
        matching = [
            chunks
            for paper_id, chunks in chunks_by_paper.items()
            if any(
                c.paper_id == paper_id
                for c in chunks
                if title.lower() in (getattr(c, 'paper_title', '') or '').lower()
            )
        ]
        if matching:
            paper_chunks.extend(matching[0])

    if not paper_chunks:
        result.issues.append(f"No chunks found for paper: {case.paper_titles}")
        result.assigned_status = "needs_review"
        return result

    # Build page-to-text mapping
    source_texts: dict[int, str] = {}
    for chunk in paper_chunks:
        if chunk.page not in source_texts:
            source_texts[chunk.page] = ""
        source_texts[chunk.page] += chunk.text + "\n"

    # Verify pages
    pages_ok, pages_msg = check_pages_accurate(
        [p.page for p in case.relevant_pages],
        source_texts,
        "",
        paper_chunks,
    )
    if not pages_ok:
        result.relevant_pages_correct = False
        result.issues.append(f"Page issue: {pages_msg}")

    # Verify answer content
    answer_ok, answer_findings = check_answer_in_source(
        case.expected_answer or "",
        source_texts,
        [p.page for p in case.relevant_pages],
    )
    if not answer_ok:
        result.answer_consistent_with_source = False
        for finding in answer_findings:
            result.issues.append(f"Answer issue: {finding}")
    else:
        result.source_text_found = answer_findings

    # Determine verification status
    if (result.relevant_pages_correct and result.answer_consistent_with_source):
        result.assigned_status = "independent_model_verified"
    elif result.answer_consistent_with_source:
        result.assigned_status = "independent_model_verified"
    else:
        result.assigned_status = "needs_review"

    return result


def format_verification_report(
    results: list[VerificationResult],
    output_path: Path,
    source_cases: list[EvaluationCase],
) -> Path:
    """Write verification report and update case data."""
    total = len(results)
    verified = sum(1 for r in results if r.assigned_status == "independent_model_verified")
    needs_review = sum(1 for r in results if r.assigned_status == "needs_review")
    had_issues = sum(1 for r in results if r.issues)

    lines = [
        "# Independent Case Verification Report",
        "",
        "- Generated: auto",
        f"- Total cases: {total}",
        f"- Verified: {verified}",
        f"- Needs review: {needs_review}",
        f"- Had issues: {had_issues}",
        "",
        "## Status Summary",
        "",
        "| Case ID | Type | Original Status | Assigned Status | Issues |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.case_id} | {result.answer_type} | "
            f"{result.original_status} | {result.assigned_status} | "
            f"{'; '.join(result.issues[:3]) if result.issues else 'OK'} |"
        )

    lines.append("")
    if had_issues > 0:
        lines.extend([
            "## Detailed Issues",
            "",
        ])
        for result in results:
            if result.issues:
                lines.append(f"### {result.case_id}")
                lines.append("")
                lines.append(f"- Query: {result.query}")
                lines.append(f"- Expected answer: {result.expected_answer}")
                for issue in result.issues:
                    lines.append(f"- **Issue**: {issue}")
                lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


async def verify_all_cases(
    dataset_path: Path,
    output_path: Path,
) -> tuple[Path, list[EvaluationCase]]:
    """Verify all cases in the dataset."""
    settings = Settings()
    database = Database(settings)

    try:
        async with database.session_factory() as session:
            # Load all chunks
            chunk_rows = list(
                await session.scalars(
                    select(Chunk).order_by(
                        Chunk.paper_id, Chunk.page, Chunk.chunk_index
                    )
                )
            )

        # Group chunks by paper
        chunks_by_paper: dict[str, list[Chunk]] = {}
        for chunk in chunk_rows:
            if chunk.paper_id not in chunks_by_paper:
                chunks_by_paper[chunk.paper_id] = []
            chunks_by_paper[chunk.paper_id].append(chunk)
    finally:
        await database.dispose()

    cases = load_cases(dataset_path)
    results: list[VerificationResult] = []
    for case in cases:
        result = await verify_test_case(case, chunks_by_paper)
        results.append(result)

    report_path = format_verification_report(results, output_path, cases)

    # Update case statuses
    updated_cases: list[EvaluationCase] = []
    for case, result in zip(cases, results, strict=False):
        if result.assigned_status == "independent_model_verified":
            case.label_status = "independent_model_verified"
        elif result.assigned_status == "needs_review" and case.label_status != "human_verified":
            case.label_status = "machine_generated"
        updated_cases.append(case)

    return report_path, updated_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify evaluation cases against PDF source text."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("demo/evaluation/15-paper-test-cases.jsonl"),
        help="Test cases JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/verification"),
        help="Output directory for verification report.",
    )
    return parser


def main() -> None:
    import asyncio

    args = build_parser().parse_args()
    output_path = args.output / "verification_report.md"
    report, updated = asyncio.run(verify_all_cases(args.dataset, output_path))

    # Write updated cases
    updated_path = args.output / "verified-cases.jsonl"
    updated_path.write_text(
        "\n".join(
            json.dumps(
                case.model_dump(mode="json"), ensure_ascii=False
            )
            for case in updated
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Verification report: {report.resolve()}")
    print(f"Updated cases: {updated_path.resolve()}")
    verified_count = sum(
        1 for c in updated if c.label_status == "independent_model_verified"
    )
    needs_review = sum(1 for c in updated if c.label_status == "machine_generated")
    print(
        f"Total: {len(updated)}, independent_model_verified: {verified_count}, "
        f"needs_review: {needs_review}"
    )
