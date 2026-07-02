import argparse
import asyncio
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Chunk, Paper
from app.db.session import Database
from app.evaluation.models import EvaluationCase, RelevantPage
from app.providers.openai_compatible import OpenAICompatibleChatProvider

AnswerType = Literal[
    "factual",
    "numeric_table",
    "architecture",
    "training",
    "ablation",
    "comparison",
    "limitation",
]


class GeneratedCase(BaseModel):
    question: str = Field(min_length=8)
    expected_answer: str = Field(min_length=3)
    pages: list[int] = Field(min_length=1, max_length=3)
    answer_type: str
    tags: list[str] = Field(default_factory=list)


class GeneratedBatch(BaseModel):
    cases: list[GeneratedCase]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate page-grounded evaluation qrels from indexed papers."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/generated/retrieval_cases.jsonl"),
    )
    parser.add_argument("--questions-per-paper", type=int, default=8)
    parser.add_argument("--max-chars-per-page", type=int, default=3600)
    return parser


def build_page_context(
    chunks: list[Chunk],
    *,
    max_chars_per_page: int,
) -> dict[int, str]:
    grouped: dict[int, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.page].append(chunk)
    pages: dict[int, str] = {}
    for page, items in sorted(grouped.items()):
        items.sort(key=lambda item: item.chunk_index)
        text = "\n".join(item.text for item in items)
        pages[page] = text[:max_chars_per_page]
    return pages


def build_prompt(
    paper: Paper,
    pages: dict[int, str],
    questions_per_paper: int,
) -> str:
    context = "\n\n".join(
        f"=== PAGE {page} ===\n{text}" for page, text in pages.items()
    )
    return f"""
Create exactly {questions_per_paper} evaluation cases for the research paper below.
Every answer must be supported entirely by the cited PDF pages.

Paper title: {paper.title}

Requirements:
- Produce an even mix of Chinese and English questions.
- Cover distinct evidence types where available: contribution, architecture,
  mechanism, training setup, datasets, quantitative tables, ablation,
  efficiency, downstream tasks, and limitations.
- At least two cases must require exact numerical or table evidence.
- Questions must identify the paper or method clearly and be answerable without
  seeing other questions.
- expected_answer must be concise, factual, and contain the exact reported values
  when the question is numerical.
- pages must contain one to three 1-based page numbers that directly support the
  complete answer.
- answer_type must be one of: factual, numeric_table, architecture, training,
  ablation, comparison, limitation.
- Do not invent missing results or infer values that are not printed.

Return one JSON object:
{{
  "cases": [
    {{
      "question": "...",
      "expected_answer": "...",
      "pages": [1],
      "answer_type": "factual",
      "tags": ["contribution"]
    }}
  ]
}}

PDF evidence:
{context}
""".strip()


def normalize_answer_type(value: str) -> AnswerType:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mechanism": "factual",
        "method": "architecture",
        "implementation": "training",
        "experimental_setup": "training",
        "dataset": "factual",
        "efficiency": "comparison",
        "downstream_task": "comparison",
        "conclusion": "factual",
    }
    allowed = {
        "factual",
        "numeric_table",
        "architecture",
        "training",
        "ablation",
        "comparison",
        "limitation",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in allowed:
        resolved = "factual"
    return cast(AnswerType, resolved)


def numeric_values(text: str) -> set[str]:
    normalized = text.replace(",", "")
    return {
        match.group(0).lower()
        for match in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", normalized)
    }


def label_status(
    case: GeneratedCase,
    pages: dict[int, str],
) -> Literal["machine_generated", "auto_checked"]:
    answer_numbers = numeric_values(case.expected_answer)
    if not answer_numbers:
        return "machine_generated"
    source = "\n".join(pages[page] for page in case.pages)
    return (
        "auto_checked"
        if answer_numbers.issubset(numeric_values(source))
        else "machine_generated"
    )


def deterministic_dev_indexes(
    paper_id: str,
    questions: list[str],
    dev_count: int,
) -> set[int]:
    ranked = sorted(
        range(len(questions)),
        key=lambda index: hashlib.sha256(
            f"{paper_id}:{questions[index]}".encode()
        ).hexdigest(),
    )
    return set(ranked[:dev_count])


def convert_cases(
    paper: Paper,
    generated: GeneratedBatch,
    pages: dict[int, str],
    questions_per_paper: int,
) -> list[EvaluationCase]:
    valid: list[GeneratedCase] = []
    seen_questions: set[str] = set()
    for item in generated.cases:
        item.pages = sorted(set(item.pages))
        if any(page not in pages for page in item.pages):
            continue
        key = " ".join(item.question.lower().split())
        if key in seen_questions:
            continue
        valid.append(item)
        seen_questions.add(key)
        if len(valid) == questions_per_paper:
            break
    if len(valid) < questions_per_paper:
        raise ValueError(
            f"{paper.title}: expected {questions_per_paper} valid cases, got {len(valid)}"
        )

    dev_count = max(2, round(questions_per_paper * 0.3))
    dev_indexes = deterministic_dev_indexes(
        paper.id,
        [item.question for item in valid],
        dev_count,
    )
    slug = re.sub(r"[^a-z0-9]+", "-", paper.title.lower()).strip("-")[:36]
    converted: list[EvaluationCase] = []
    for index, item in enumerate(valid, start=1):
        converted.append(
            EvaluationCase(
                case_id=f"{slug}-{paper.id[:8]}-{index:02d}",
                query=item.question,
                paper_titles=[paper.title],
                relevant_pages=[
                    RelevantPage(paper_title=paper.title, page=page)
                    for page in item.pages
                ],
                expected_answer=item.expected_answer,
                answer_type=normalize_answer_type(item.answer_type),
                split="dev" if index - 1 in dev_indexes else "test",
                label_status=label_status(item, pages),
                tags=sorted(set(item.tags)),
            )
        )
    return converted


async def generate(
    output: Path,
    questions_per_paper: int,
    max_chars_per_page: int,
) -> list[EvaluationCase]:
    settings = Settings()
    database = Database(settings)
    provider = OpenAICompatibleChatProvider(
        settings.chat_api_key,
        settings.chat_base_url,
        settings.chat_model,
        settings.chat_thinking,
    )
    try:
        async with database.session_factory() as session:
            papers = list(
                await session.scalars(
                    select(Paper)
                    .where(
                        Paper.status == "ready",
                        Paper.index_status == "ready",
                    )
                    .order_by(Paper.created_at)
                )
            )
            chunk_rows = list(
                await session.scalars(
                    select(Chunk).order_by(
                        Chunk.paper_id,
                        Chunk.page,
                        Chunk.chunk_index,
                    )
                )
            )
    finally:
        await database.dispose()

    chunks_by_paper: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunk_rows:
        chunks_by_paper[chunk.paper_id].append(chunk)

    output.parent.mkdir(parents=True, exist_ok=True)
    all_cases = await asyncio.to_thread(load_existing_cases, output)
    completed_counts: dict[str, int] = defaultdict(int)
    for case in all_cases:
        for title in case.paper_titles:
            completed_counts[title] += 1
    for paper_number, paper in enumerate(papers, start=1):
        if completed_counts[paper.title] >= questions_per_paper:
            print(
                f"[{paper_number}/{len(papers)}] {paper.title}: "
                f"resume existing {completed_counts[paper.title]} cases"
            )
            continue
        all_cases = [
            case for case in all_cases if paper.title not in case.paper_titles
        ]
        pages = build_page_context(
            chunks_by_paper[paper.id],
            max_chars_per_page=max_chars_per_page,
        )
        prompt = build_prompt(paper, pages, questions_per_paper)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                payload = await provider.structured(prompt, "evaluation_cases")
                generated = GeneratedBatch.model_validate(payload)
                cases = convert_cases(
                    paper,
                    generated,
                    pages,
                    questions_per_paper,
                )
                all_cases.extend(cases)
                await asyncio.to_thread(write_cases, output, all_cases)
                print(
                    f"[{paper_number}/{len(papers)}] {paper.title}: "
                    f"{len(cases)} cases"
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    await asyncio.sleep(attempt)
        else:
            raise RuntimeError(f"Failed to generate cases for {paper.title}") from last_error
    await asyncio.to_thread(
        write_cases,
        output.with_name("dev_cases.jsonl"),
        [case for case in all_cases if case.split == "dev"],
    )
    await asyncio.to_thread(
        write_cases,
        output.with_name("test_cases.jsonl"),
        [case for case in all_cases if case.split == "test"],
    )
    return all_cases


def load_existing_cases(output: Path) -> list[EvaluationCase]:
    if not output.exists():
        return []
    return [
        EvaluationCase.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_cases(output: Path, cases: list[EvaluationCase]) -> None:
    output.write_text(
        "\n".join(
            json.dumps(case.model_dump(mode="json"), ensure_ascii=False)
            for case in cases
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    cases = asyncio.run(
        generate(
            args.output,
            args.questions_per_paper,
            args.max_chars_per_page,
        )
    )
    print(f"Generated {len(cases)} cases at {args.output.resolve()}")


if __name__ == "__main__":
    main()
