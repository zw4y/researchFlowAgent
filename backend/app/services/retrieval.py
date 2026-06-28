from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Chunk, Paper
from app.providers.base import EmbeddingProvider, Evidence
from app.rag.vector_store import VectorRecord, VectorStore


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def search(self, query: str, paper_ids: list[str]) -> list[Evidence]:
        query_vector = (await self.embedding_provider.embed([query]))[0]
        hits = await self.vector_store.search(
            query_vector,
            paper_ids,
            self.settings.retrieval_top_k,
        )
        return [
            Evidence(
                chunk_id=str(hit.payload["chunk_id"]),
                paper_id=str(hit.payload["paper_id"]),
                paper_title=str(hit.payload["paper_title"]),
                page=int(hit.payload["page"]),
                text=str(hit.payload["text"]),
                score=hit.score,
            )
            for hit in hits
            if hit.score >= self.settings.retrieval_score_threshold
        ]

    async def rehydrate_memory_store(self) -> None:
        if self.vector_store.name != "memory":
            return
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(Chunk, Paper.title)
                    .join(Paper, Paper.id == Chunk.paper_id)
                    .where(Paper.status == "ready")
                )
            ).all()
        if not rows:
            return
        texts = [chunk.text for chunk, _ in rows]
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 64):
            embeddings.extend(await self.embedding_provider.embed(texts[start : start + 64]))
        records = [
            VectorRecord(
                id=chunk.vector_id,
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "paper_id": chunk.paper_id,
                    "paper_title": title,
                    "page": chunk.page,
                    "text": chunk.text,
                },
            )
            for (chunk, title), vector in zip(rows, embeddings, strict=True)
        ]
        await self.vector_store.upsert(records)
