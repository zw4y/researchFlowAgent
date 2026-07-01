"""Answer-level evaluation runner: generates answers for four schemes and evaluates them.

Schemes:
  A. DeepSeek closed-book (no paper, no RAG)
  B. DeepSeek full-paper context (all indexed chunks)
  C. Vector-only RAG (Qdrant Top 6, no rerank)
  D. ResearchFlow full pipeline (Vector Top 20 + rerank Top 6)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import tiktoken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Chunk, Paper
from app.evaluation.answer_metrics import (
    aggregate_answer_metrics,
    compute_bootstrap_ci,
    evaluate_answer,
    format_markdown_report,
    stratify_by_paper,
    stratify_by_type,
)
from app.evaluation.answer_models import (
    AnswerCase,
    AnswerMetrics,
    AnswerReport,
    AnswerScheme,
    GeneratedAnswer,
)
from app.evaluation.models import EvaluationCase
from app.providers.base import Evidence, RerankProvider
from app.providers.openai_compatible import OpenAICompatibleChatProvider
from app.rag.index_profile import IndexProfile
from app.rag.vector_store import LlamaIndexVectorStore

logger = logging.getLogger(__name__)

# Cost per 1K tokens (CNY) - DeepSeek V4 Flash pricing
# https://api-docs.deepseek.com/quick_start/pricing
_INPUT_COST_PER_1K = 0.001  # ~0.001 CNY per 1K input tokens
_OUTPUT_COST_PER_1K = 0.002  # ~0.002 CNY per 1K output tokens


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost in CNY."""
    return (input_tokens / 1000) * _INPUT_COST_PER_1K + (output_tokens / 1000) * _OUTPUT_COST_PER_1K


class AnswerEvaluationRunner:
    """Runs answer-level evaluation across four comparison schemes."""

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: LlamaIndexVectorStore,
        rerank_provider: RerankProvider,
        profile: IndexProfile,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.rerank_provider = rerank_provider
        self.profile = profile
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chat_provider = OpenAICompatibleChatProvider(
            settings.chat_api_key,
            settings.chat_base_url,
            settings.chat_model,
            settings.chat_thinking,
        )

    async def run(
        self,
        cases: list[EvaluationCase],
        dataset_path: Path,
        *,
        schemes: list[AnswerScheme] | None = None,
    ) -> AnswerReport:
        """Run answer-level evaluation for specified schemes."""
        if schemes is None:
            schemes = ["closed_book", "full_paper", "vector_rag", "researchflow"]

        answer_cases = [self._to_answer_case(case) for case in cases]
        report = AnswerReport(
            dataset_path=str(dataset_path),
            schemes=list(schemes),
            case_count=len(answer_cases),
        )

        for scheme in schemes:
            logger.info("Running scheme: %s (%d cases)", scheme, len(answer_cases))
            scheme_cases = await self._run_scheme(answer_cases, scheme)
            # Update metric results
            for acase in scheme_cases:
                for existing in report.cases:
                    if existing.case_id == acase.case_id:
                        if scheme in acase.answers:
                            existing.answers[scheme] = acase.answers[scheme]
                        if scheme in acase.metrics:
                            existing.metrics[scheme] = acase.metrics[scheme]
                        break
                else:
                    report.cases.append(acase)

        # Compute summaries
        all_metrics: dict[AnswerScheme, list[AnswerMetrics]] = {
            s: [] for s in schemes
        }
        for case in report.cases:
            for scheme in schemes:
                if scheme in case.metrics:
                    all_metrics[scheme].append(case.metrics[scheme])

        for scheme in schemes:
            report.summaries[scheme] = aggregate_answer_metrics(
                all_metrics[scheme], scheme
            )

        # Stratified results
        for scheme in schemes:
            report.by_answer_type[scheme] = stratify_by_type(report.cases, scheme)
            report.by_paper[scheme] = stratify_by_paper(report.cases, scheme)

        # Bootstrap CI (researchflow vs vector_rag)
        if "researchflow" in schemes and "vector_rag" in schemes:
            rf_metrics = all_metrics.get("researchflow", [])
            vr_metrics = all_metrics.get("vector_rag", [])
            if rf_metrics and vr_metrics:
                for key in [
                    "answer_correctness",
                    "faithfulness",
                    "hallucination_rate",
                ]:
                    report.bootstrap_ci[key] = compute_bootstrap_ci(
                        vr_metrics, rf_metrics, key
                    )

        # Collect failures
        report.failures = self._analyze_failures(report.cases, schemes)

        return report

    async def _run_scheme(
        self,
        cases: list[AnswerCase],
        scheme: AnswerScheme,
    ) -> list[AnswerCase]:
        """Generate answers for all cases using one scheme."""
        updated: list[AnswerCase] = []
        for case in cases:
            updated.append(
                await self._generate_and_evaluate(case, scheme)
            )
        return updated

    async def _generate_and_evaluate(
        self,
        case: AnswerCase,
        scheme: AnswerScheme,
    ) -> AnswerCase:
        """Generate answer for one case with one scheme, then evaluate."""
        try:
            if scheme == "closed_book":
                answer = await self._closed_book(case)
            elif scheme == "full_paper":
                answer = await self._full_paper(case)
            elif scheme == "vector_rag":
                answer = await self._vector_rag(case)
            elif scheme == "researchflow":
                answer = await self._researchflow(case)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            case.answers[scheme] = answer

            # Get evidence texts for faithfulness check
            evidence_texts = await self._get_evidence_texts(case)

            metrics = evaluate_answer(
                case_id=case.case_id,
                scheme=scheme,
                answer_type=case.answer_type,
                generated_answer=answer.answer_text,
                expected_answer=case.expected_answer or "",
                relevant_pages=case.relevant_pages,
                evidence_texts=evidence_texts,
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                full_context_tokens=answer.input_tokens,
                rag_context_tokens=answer.input_tokens,
                api_cost_cny=answer.api_cost_cny,
                latency_ms=answer.latency_ms,
            )
            case.metrics[scheme] = metrics

        except Exception as exc:
            logger.error("Failed to evaluate case %s scheme %s: %s",
                         case.case_id, scheme, exc)
            case.answers[scheme] = GeneratedAnswer(
                case_id=case.case_id,
                scheme=scheme,
                answer_text=f"[error] {exc}",
                tool_call_success=False,
                grounding_status="ungrounded",
            )

        return case

    async def _closed_book(self, case: AnswerCase) -> GeneratedAnswer:
        """Scheme A: DeepSeek closed-book, no paper evidence."""
        started = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Answer the question based on your "
                    "own knowledge. Do not fabricate citations. If you don't know, "
                    "say you cannot answer."
                ),
            },
            {"role": "user", "content": case.query},
        ]
        text = await self.chat_provider.chat(messages)
        latency = round((time.perf_counter() - started) * 1000)
        input_tokens = len(self.encoding.encode(str(messages)))
        output_tokens = len(self.encoding.encode(text))
        return GeneratedAnswer(
            case_id=case.case_id,
            scheme="closed_book",
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            api_cost_cny=estimate_cost(input_tokens, output_tokens),
            tool_call_success=True,
            evidence_count=0,
            has_citations=False,
            grounding_status="ungrounded",
        )

    async def _full_paper(self, case: AnswerCase) -> GeneratedAnswer:
        """Scheme B: Full-paper context (all indexed chunks)."""
        paper_ids = await self._resolve_paper_ids(case)
        full_text = await self._load_full_text(paper_ids)
        input_tokens = len(self.encoding.encode(full_text + case.query))
        started = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ResearchFlow, a careful research assistant. Answer using "
                    "the paper text provided. Cite page numbers when possible. "
                    "If the paper text does not contain the answer, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Paper content:\n{full_text}\n\nQuestion:\n{case.query}",
            },
        ]
        text = await self.chat_provider.chat(messages)
        latency = round((time.perf_counter() - started) * 1000)
        output_tokens = len(self.encoding.encode(text))
        return GeneratedAnswer(
            case_id=case.case_id,
            scheme="full_paper",
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            api_cost_cny=estimate_cost(input_tokens, output_tokens),
            tool_call_success=True,
            evidence_count=0,
            has_citations="[P" in text or "page" in text.lower(),
            grounding_status="partially_grounded",
        )

    async def _vector_rag(self, case: AnswerCase) -> GeneratedAnswer:
        """Scheme C: Vector-only RAG (Top 6, no rerank).

        Uses Qdrant directly without reranking, then sends to DeepSeek.
        """
        paper_ids = await self._resolve_paper_ids(case)
        started = time.perf_counter()

        # Direct vector retrieval
        nodes = await self.vector_store.retrieve(
            case.query,
            paper_ids,
            self.settings.rerank_top_n,
        )
        retrieval_latency = round((time.perf_counter() - started) * 1000)

        # Build evidence
        evidence: list[Evidence] = []
        for node in nodes[: self.settings.rerank_top_n]:
            metadata = node.node.metadata
            evidence.append(
                Evidence(
                    chunk_id=str(metadata["chunk_id"]),
                    paper_id=str(metadata["paper_id"]),
                    paper_title=str(metadata["paper_title"]),
                    page=int(metadata["page"]),
                    text=node.node.get_content(),
                    score=float(node.score or 0.0),
                    retrieval_status="vector",
                )
            )

        rag_text = "\n".join(
            f"[P{idx}] {item.paper_title}, page {item.page}: {item.text}"
            for idx, item in enumerate(evidence, start=1)
        )
        input_tokens = len(self.encoding.encode(rag_text + case.query))

        llm_started = time.perf_counter()
        text = await self.chat_provider.synthesize(case.query, evidence, [], [])
        llm_latency = round((time.perf_counter() - llm_started) * 1000)

        output_tokens = len(self.encoding.encode(text))
        return GeneratedAnswer(
            case_id=case.case_id,
            scheme="vector_rag",
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=retrieval_latency + llm_latency,
            api_cost_cny=estimate_cost(input_tokens, output_tokens),
            tool_call_success=True,
            evidence_count=len(evidence),
            has_citations="[P" in text,
            grounding_status="grounded" if evidence else "ungrounded",
        )

    async def _researchflow(self, case: AnswerCase) -> GeneratedAnswer:
        """Scheme D: ResearchFlow full pipeline (Top 20 + rerank + synthesize)."""

        # Create a temporary retrieval service (or use the one from container)
        # We use the one we already have - but we need to access it differently
        paper_ids = await self._resolve_paper_ids(case)
        started = time.perf_counter()

        # Use vector store directly for candidate retrieval
        candidates = await self.vector_store.retrieve(
            case.query,
            paper_ids,
            self.settings.retrieval_candidates,
        )

        # Rerank
        if self.rerank_provider.enabled and candidates:
            try:
                results = await self.rerank_provider.rerank(
                    case.query,
                    [item.node.get_content() for item in candidates],
                    self.settings.rerank_top_n,
                )
                reranked_nodes = []
                for result in results:
                    candidate = candidates[result.index]
                    reranked_nodes.append(candidate)
                selected = reranked_nodes[: self.settings.rerank_top_n]
                retrieval_status = "reranked"
            except Exception:
                selected = candidates[: self.settings.rerank_top_n]
                retrieval_status = "rerank_fallback"
        else:
            selected = candidates[: self.settings.rerank_top_n]
            retrieval_status = "vector"

        retrieval_latency = round((time.perf_counter() - started) * 1000)

        evidence: list[Evidence] = []
        for node in selected:
            metadata = node.node.metadata
            evidence.append(
                Evidence(
                    chunk_id=str(metadata["chunk_id"]),
                    paper_id=str(metadata["paper_id"]),
                    paper_title=str(metadata["paper_title"]),
                    page=int(metadata["page"]),
                    text=node.node.get_content(),
                    score=float(node.score or 0.0),
                    retrieval_status=retrieval_status,  # type: ignore[arg-type]
                )
            )

        rag_text = "\n".join(
            f"[P{idx}] {item.paper_title}, page {item.page}: {item.text}"
            for idx, item in enumerate(evidence, start=1)
        )
        input_tokens = len(self.encoding.encode(rag_text + case.query))

        llm_started = time.perf_counter()
        text = await self.chat_provider.synthesize(case.query, evidence, [], [])
        llm_latency = round((time.perf_counter() - llm_started) * 1000)

        output_tokens = len(self.encoding.encode(text))
        return GeneratedAnswer(
            case_id=case.case_id,
            scheme="researchflow",
            answer_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=retrieval_latency + llm_latency,
            api_cost_cny=estimate_cost(input_tokens, output_tokens),
            tool_call_success=True,
            evidence_count=len(evidence),
            has_citations="[P" in text,
            retrieval_status=retrieval_status,
            grounding_status="grounded" if evidence else "ungrounded",
        )

    async def _resolve_paper_ids(self, case: AnswerCase) -> list[str]:
        """Resolve paper titles or IDs to database IDs."""
        async with self.session_factory() as session:
            statement = select(Paper.id).where(
                Paper.status == "ready",
                Paper.index_status == "ready",
                Paper.index_profile == self.profile.profile_id,
            )
            if case.paper_ids:
                statement = statement.where(Paper.id.in_(case.paper_ids))
            elif case.paper_titles:
                statement = statement.where(Paper.title.in_(case.paper_titles))
            return list(await session.scalars(statement))

    async def _load_full_text(self, paper_ids: list[str]) -> str:
        """Load full text of all chunks for given paper IDs."""
        async with self.session_factory() as session:
            chunks = list(
                await session.scalars(
                    select(Chunk)
                    .where(
                        Chunk.paper_id.in_(paper_ids),
                        Chunk.index_profile == self.profile.profile_id,
                    )
                    .order_by(Chunk.paper_id, Chunk.page, Chunk.chunk_index)
                )
            )
        if not chunks:
            return ""

        parts: list[str] = []
        current_title = ""
        for chunk in chunks:
            if chunk.paper_id != current_title:
                current_title = chunk.paper_id
                parts.append(f"\n--- Paper {current_title} ---")
            parts.append(f"[Page {chunk.page}] {chunk.text}")
        return "\n".join(parts)

    async def _get_evidence_texts(self, case: AnswerCase) -> list[str]:
        """Get evidence texts for a case (from database)."""
        paper_ids = await self._resolve_paper_ids(case)
        async with self.session_factory() as session:
            chunks = list(
                await session.scalars(
                    select(Chunk)
                    .where(
                        Chunk.paper_id.in_(paper_ids),
                        Chunk.index_profile == self.profile.profile_id,
                    )
                    .limit(50)
                )
            )
        return [chunk.text for chunk in chunks]

    @staticmethod
    def _to_answer_case(eval_case: EvaluationCase) -> AnswerCase:
        """Convert an EvaluationCase to an AnswerCase."""
        return AnswerCase(
            case_id=eval_case.case_id,
            query=eval_case.query,
            paper_titles=eval_case.paper_titles,
            paper_ids=eval_case.paper_ids,
            relevant_pages=[
                {"paper_title": p.paper_title, "page": p.page}
                for p in eval_case.relevant_pages
            ],
            expected_answer=eval_case.expected_answer,
            answer_type=eval_case.answer_type,
            split=eval_case.split,
            label_status=eval_case.label_status,
        )

    @staticmethod
    def _analyze_failures(
        cases: list[AnswerCase],
        schemes: list[AnswerScheme],
    ) -> list[dict[str, Any]]:
        """Identify failure cases for analysis."""
        failures: list[dict[str, Any]] = []
        for case in cases:
            for scheme in schemes:
                metrics = case.metrics.get(scheme)
                if not metrics:
                    continue
                # Numeric exact match failures
                if metrics.numeric_values_in_reference > 0 and not metrics.numeric_exact_match:
                    failures.append({
                        "case_id": case.case_id,
                        "scheme": scheme,
                        "type": "numeric_mismatch",
                        "description": (
                            f"Expected {metrics.numeric_values_in_reference} numeric values, "
                            f"found {metrics.numeric_values_in_answer} in answer"
                        ),
                    })
                # Hallucination
                if metrics.hallucination_rate > 0.5:
                    failures.append({
                        "case_id": case.case_id,
                        "scheme": scheme,
                        "type": "hallucination",
                        "description": (
                            f"Hallucination rate: {metrics.hallucination_rate:.1%}"
                        ),
                    })
                # Low faithfulness
                if metrics.faithfulness < 0.3:
                    failures.append({
                        "case_id": case.case_id,
                        "scheme": scheme,
                        "type": "low_faithfulness",
                        "description": (
                            f"Faithfulness: {metrics.faithfulness:.1%}"
                        ),
                    })
                # Citation issues
                if metrics.citation_precision < 0.5 and metrics.citation_recall < 0.5:
                    failures.append({
                        "case_id": case.case_id,
                        "scheme": scheme,
                        "type": "citation_issue",
                        "description": (
                            f"Citation precision: {metrics.citation_precision:.1%}, "
                            f"recall: {metrics.citation_recall:.1%}"
                        ),
                    })
        return failures


def write_answer_report(
    report: AnswerReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write answer evaluation report to JSON and Markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "answer_report.json"
    markdown_path = output_dir / "answer_report.md"

    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        format_markdown_report(report), encoding="utf-8"
    )
    return json_path, markdown_path
