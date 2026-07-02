import argparse
import asyncio
from pathlib import Path

from app.container import AppContainer
from app.core.config import Settings
from app.evaluation.runner import RetrievalEvaluationRunner, load_cases, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ResearchFlow retrieval quality and context-token efficiency."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("demo/evaluation/retrieval_cases.jsonl"),
        help="JSONL qrels dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/latest"),
        help="Directory for report.json and report.md.",
    )
    return parser


async def run(dataset: Path, output: Path) -> tuple[Path, Path]:
    settings = Settings()
    container = AppContainer(settings)
    await container.start()
    try:
        runner = RetrievalEvaluationRunner(
            settings,
            container.database.session_factory,
            container.retrieval_service,
            container.index_profile,
        )
        report = await runner.run(load_cases(dataset), dataset)
        return write_report(report, output)
    finally:
        await container.close()


def main() -> None:
    args = build_parser().parse_args()
    json_path, markdown_path = asyncio.run(run(args.dataset, args.output))
    print(f"JSON report: {json_path.resolve()}")
    print(f"Markdown report: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()
