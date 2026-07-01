import logging

from llama_index.core.schema import NodeWithScore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Paper
from app.providers.base import Evidence, RerankProvider
from app.rag.index_profile import IndexProfile
from app.rag.vector_store import LlamaIndexVectorStore

logger = logging.getLogger(__name__)

_TABLE_QUERY_TERMS = (
    "指标",
    "数据集",
    "数值",
    "性能",
    "表格",
    "排名",
    "metric",
    "dataset",
    "score",
    "performance",
    "table",
    "benchmark",
    "accuracy",
)


class RetrievalService:
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

    async def search(self, query: str, paper_ids: list[str]) -> list[Evidence]:
        allowed_ids = await self._ready_paper_ids(paper_ids)
        if not allowed_ids:
            return []
        candidate_limit = self.settings.retrieval_candidates
        if self._is_table_query(query):
            candidate_limit = max(candidate_limit, 40)
        nodes = await self.vector_store.retrieve(
            query,
            allowed_ids,
            candidate_limit,
        )
        candidates = [
            node
            for node in nodes
            if float(node.score or 0.0) >= self.settings.retrieval_score_threshold
        ]
        for candidate in candidates:
            candidate.node.metadata["vector_score"] = float(candidate.score or 0.0)
        selected, retrieval_status = await self._rerank(query, candidates)
        selected = self._prioritize_table_summaries(query, candidates, selected)
        return [self._to_evidence(node, retrieval_status) for node in selected]

    @staticmethod
    def _is_table_query(query: str) -> bool:
        lowered = query.lower()
        return any(term in lowered for term in _TABLE_QUERY_TERMS)

    def _prioritize_table_summaries(
        self,
        query: str,
        candidates: list[NodeWithScore],
        selected: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        if not self._is_table_query(query):
            return selected
        summaries = [
            candidate
            for candidate in candidates
            if "[Table key rows]" in candidate.node.get_content()
        ]
        if not summaries:
            return selected
        result: list[NodeWithScore] = []
        seen: set[str] = set()
        for candidate in [*summaries, *selected]:
            node_id = candidate.node.node_id
            if node_id in seen:
                continue
            result.append(candidate)
            seen.add(node_id)
            if len(result) >= self.settings.rerank_top_n:
                break
        return result


    async def _ready_paper_ids(self, requested_ids: list[str]) -> list[str]:
        async with self.session_factory() as session:
            statement = select(Paper.id).where(
                Paper.status == "ready",
                Paper.index_status == "ready",
                Paper.index_profile == self.profile.profile_id,
            )
            if requested_ids:
                statement = statement.where(Paper.id.in_(requested_ids))
            return list(await session.scalars(statement))

    async def _rerank(
        self,
        query: str,
        candidates: list[NodeWithScore],
    ) -> tuple[list[NodeWithScore], str]:
        top_n = min(self.settings.rerank_top_n, len(candidates))
        if top_n == 0:
            return [], "vector"
        if not self.rerank_provider.enabled:
            return candidates[:top_n], "vector"
        try:
            results = await self.rerank_provider.rerank(
                query,
                [item.node.get_content() for item in candidates],
                top_n,
            )
            reranked: list[NodeWithScore] = []
            for result in results:
                candidate = candidates[result.index]
                reranked.append(NodeWithScore(node=candidate.node, score=result.score))
            return reranked, "reranked"
        except Exception as exc:
            logger.warning("Rerank failed; using vector order: %s", exc)
            return candidates[:top_n], "rerank_fallback"

    @staticmethod
    def _to_evidence(node: NodeWithScore, retrieval_status: str) -> Evidence:
        metadata = node.node.metadata
        vector_score = metadata.get("vector_score")
        return Evidence(
            chunk_id=str(metadata["chunk_id"]),
            paper_id=str(metadata["paper_id"]),
            paper_title=str(metadata["paper_title"]),
            page=int(metadata["page"]),
            text=node.node.get_content(),
            score=float(node.score or 0.0),
            retrieval_status=retrieval_status,  # type: ignore[arg-type]
            vector_score=float(vector_score) if vector_score is not None else None,
        )
