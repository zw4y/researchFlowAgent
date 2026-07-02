import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import tiktoken
from llama_index.core.schema import NodeWithScore
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Chunk, Paper
from app.evaluation.metrics import aggregate_results, evaluate_ranking
from app.evaluation.models import CaseResult, EvaluationCase, EvaluationReport
from app.providers.base import RetrievalStatus
from app.rag.index_profile import IndexProfile
from app.services.retrieval import RetrievalService


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: {exc}"
            ) from exc
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


class RetrievalEvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        retrieval_service: RetrievalService,
        profile: IndexProfile,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.retrieval_service = retrieval_service
        self.profile = profile
        self.encoding = tiktoken.get_encoding("cl100k_base")

    async def run(
        self,
        cases: list[EvaluationCase],
        dataset_path: Path,
    ) -> EvaluationReport:
        results = [await self._run_case(case) for case in cases]
        return EvaluationReport(
            generated_at=datetime.now(UTC),
            dataset_path=str(dataset_path),
            index_profile=self.profile.profile_id,
            embedding_model=self.profile.model,
            rerank_model=self.settings.rerank_model,
            top_k=self.settings.rerank_top_n,
            summary=aggregate_results(results),
            cases=results,
        )

    async def _run_case(self, case: EvaluationCase) -> CaseResult:
        paper_ids = await self._resolve_paper_ids(case)
        trace = await self.retrieval_service.trace(
            case.query,
            paper_ids,
            candidate_limit=self.settings.retrieval_candidates,
        )
        relevant_keys = self._relevant_keys(case)
        candidate_keys = self._node_keys(trace.candidates, case)
        vector_keys = self._node_keys(trace.vector_selected, case)
        reranked_keys = self._node_keys(trace.selected, case)
        top_k = self.settings.rerank_top_n
        full_context_tokens = await self._full_context_tokens(paper_ids)
        return CaseResult(
            case_id=case.case_id,
            query=case.query,
            candidate_k=self.settings.retrieval_candidates,
            vector=evaluate_ranking(vector_keys, relevant_keys, k=top_k),
            reranked=evaluate_ranking(reranked_keys, relevant_keys, k=top_k),
            candidate_recall=evaluate_ranking(
                candidate_keys,
                relevant_keys,
                k=max(1, len(candidate_keys)),
            ).recall,
            full_context_tokens=full_context_tokens,
            vector_context_tokens=self._context_tokens(trace.vector_selected),
            reranked_context_tokens=self._context_tokens(trace.selected),
            vector_latency_ms=trace.vector_latency_ms,
            rerank_latency_ms=trace.rerank_latency_ms,
            retrieval_status=cast(RetrievalStatus, trace.retrieval_status),
            vector_keys=vector_keys,
            reranked_keys=reranked_keys,
        )

    async def _resolve_paper_ids(self, case: EvaluationCase) -> list[str]:
        async with self.session_factory() as session:
            statement = select(Paper.id, Paper.title).where(
                Paper.status == "ready",
                Paper.index_status == "ready",
                Paper.index_profile == self.profile.profile_id,
            )
            if case.paper_ids:
                statement = statement.where(Paper.id.in_(case.paper_ids))
            else:
                statement = statement.where(Paper.title.in_(case.paper_titles))
            rows = list((await session.execute(statement)).all())

        expected = set(case.paper_ids or case.paper_titles)
        found = {row.id if case.paper_ids else row.title for row in rows}
        missing = sorted(expected - found)
        if missing:
            raise ValueError(
                f"Case {case.case_id} references missing or stale papers for "
                f"profile {self.profile.profile_id}: {missing}"
            )
        return [row.id for row in rows]

    async def _full_context_tokens(self, paper_ids: list[str]) -> int:
        async with self.session_factory() as session:
            total = await session.scalar(
                select(func.sum(Chunk.token_count)).where(
                    Chunk.paper_id.in_(paper_ids),
                    Chunk.index_profile == self.profile.profile_id,
                )
            )
        return int(total or 0)

    @staticmethod
    def _relevant_keys(case: EvaluationCase) -> set[str]:
        if case.relevant_chunk_ids:
            return {f"chunk:{chunk_id}" for chunk_id in case.relevant_chunk_ids}
        return {page.key for page in case.relevant_pages}

    @staticmethod
    def _node_keys(
        nodes: list[NodeWithScore],
        case: EvaluationCase,
    ) -> list[str]:
        if case.relevant_chunk_ids:
            return [f"chunk:{node.node.metadata['chunk_id']}" for node in nodes]
        return [
            f"{node.node.metadata['paper_title']}:{node.node.metadata['page']}"
            for node in nodes
        ]

    def _context_tokens(self, nodes: list[NodeWithScore]) -> int:
        return sum(len(self.encoding.encode(node.node.get_content())) for node in nodes)


def write_report(report: EvaluationReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def format_markdown(report: EvaluationReport) -> str:
    summary = report.summary

    def percent(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        "# ResearchFlow Retrieval Evaluation",
        "",
        f"- Generated: `{report.generated_at.isoformat()}`",
        f"- Index profile: `{report.index_profile}`",
        f"- Embedding: `{report.embedding_model}`",
        f"- Reranker: `{report.rerank_model}`",
        f"- Cases: `{summary.case_count}`",
        "",
        "## Summary",
        "",
        "| Metric | Vector Top-K | Reranked Top-K | Delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Precision@{report.top_k} | {percent(summary.vector_precision)} | "
            f"{percent(summary.reranked_precision)} | "
            f"{percent(summary.precision_delta)} |"
        ),
        (
            f"| Recall@{report.top_k} | {percent(summary.vector_recall)} | "
            f"{percent(summary.reranked_recall)} | {percent(summary.recall_delta)} |"
        ),
        (
            f"| MRR@{report.top_k} | {summary.vector_mrr:.4f} | "
            f"{summary.reranked_mrr:.4f} | {summary.mrr_delta:+.4f} |"
        ),
        (
            f"| Context token savings vs full paper | "
            f"{percent(summary.vector_token_savings)} | "
            f"{percent(summary.reranked_token_savings)} | "
            f"{percent(summary.reranked_token_savings - summary.vector_token_savings)} |"
        ),
        "",
        f"- Candidate Recall@{report.cases[0].candidate_k}: "
        f"**{percent(summary.candidate_recall)}**",
        f"- Rerank API success rate: **{percent(summary.rerank_success_rate)}**",
        f"- Average vector latency: **{summary.average_vector_latency_ms:.1f} ms**",
        f"- Average rerank latency: **{summary.average_rerank_latency_ms:.1f} ms**",
        "",
        "Token savings measures paper-context input only. It does not claim a reduction "
        "in generated output tokens or billing without provider usage records.",
        "",
        "## Cases",
        "",
        "| Case | Vector P/R/MRR | Reranked P/R/MRR | Full tokens | RAG tokens | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in report.cases:
        lines.append(
            f"| {item.case_id} | "
            f"{item.vector.precision:.3f}/{item.vector.recall:.3f}/"
            f"{item.vector.reciprocal_rank:.3f} | "
            f"{item.reranked.precision:.3f}/{item.reranked.recall:.3f}/"
            f"{item.reranked.reciprocal_rank:.3f} | "
            f"{item.full_context_tokens} | {item.reranked_context_tokens} | "
            f"{item.retrieval_status} |"
        )
    lines.append("")
    return "\n".join(lines)
