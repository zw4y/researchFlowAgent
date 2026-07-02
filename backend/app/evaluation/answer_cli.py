"""CLI for answer-level evaluation across four comparison schemes.

Usage:
  python -m app.evaluation.answer_cli \\
    --dataset demo/evaluation/15-paper-test-cases.jsonl \\
    --output data/evaluations/answer \\
    --schemes closed_book full_paper vector_rag researchflow

Or run specific schemes by name:
  python -m app.evaluation.answer_cli --schemes researchflow
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.container import AppContainer
from app.core.config import Settings
from app.evaluation.answer_models import AnswerScheme
from app.evaluation.answer_runner import (
    AnswerEvaluationRunner,
    write_answer_report,
)
from app.evaluation.runner import load_cases

logger = logging.getLogger(__name__)

_KNOWN_SCHEMES: list[AnswerScheme] = [
    "closed_book",
    "full_paper",
    "vector_rag",
    "researchflow",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer-level evaluation for ResearchFlow."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("demo/evaluation/15-paper-test-cases.jsonl"),
        help="Test cases JSONL file (default: 15-paper-test-cases).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/answer"),
        help="Output directory for reports (default: data/evaluations/answer).",
    )
    parser.add_argument(
        "--schemes",
        type=str,
        nargs="+",
        choices=_KNOWN_SCHEMES,
        default=None,
        help="Schemes to evaluate. Default: all four.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N cases for testing (default: all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


async def run_evaluation(
    dataset: Path,
    output: Path,
    schemes: list[AnswerScheme] | None,
    limit: int,
) -> tuple[Path, Path]:
    """Run answer-level evaluation."""
    settings = Settings()
    container = AppContainer(settings)
    await container.start()

    try:
        runner = AnswerEvaluationRunner(
            settings,
            container.database.session_factory,
            container.vector_store,
            container.rerank_provider,
            container.index_profile,
        )

        all_cases = load_cases(dataset)
        if limit > 0:
            all_cases = all_cases[:limit]

        # Filter to test split only
        test_cases = [c for c in all_cases if c.split == "test"]
        if not test_cases:
            test_cases = all_cases

        logger.info("Loaded %d test cases, running schemes: %s",
                     len(test_cases), schemes or _KNOWN_SCHEMES)

        report = await runner.run(
            test_cases,
            dataset,
            schemes=schemes,
        )
        return write_answer_report(report, output)
    finally:
        await container.close()


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    json_path, markdown_path = asyncio.run(
        run_evaluation(
            args.dataset,
            args.output,
            args.schemes,
            args.limit,
        )
    )
    print(f"JSON report: {json_path.resolve()}")
    print(f"Markdown report: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()
