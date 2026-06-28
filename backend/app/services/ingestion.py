import asyncio
import logging

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import AppError
from app.db.models import Chunk, IngestionJob, Paper, new_id
from app.providers.base import EmbeddingProvider
from app.rag.pdf import chunk_pages, parse_pdf
from app.rag.vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
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

    async def process(self, paper_id: str, job_id: str) -> None:
        try:
            await self._set_status(paper_id, job_id, "processing", 5)
            async with self.session_factory() as session:
                paper = await session.get(Paper, paper_id)
                if paper is None:
                    raise AppError("论文不存在。", status_code=404, code="paper_not_found")
                file_path = self.settings.upload_dir / paper.stored_filename
                original_title = paper.title

            parsed = await asyncio.to_thread(parse_pdf, file_path, self.settings.max_pdf_pages)
            chunks = chunk_pages(
                parsed.pages,
                max_tokens=self.settings.chunk_size_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
            )
            await self._set_status(paper_id, job_id, "processing", 25)

            vectors: list[list[float]] = []
            for start in range(0, len(chunks), 64):
                batch = chunks[start : start + 64]
                vectors.extend(await self.embedding_provider.embed([item.text for item in batch]))
                progress = 25 + int(55 * min(start + 64, len(chunks)) / max(len(chunks), 1))
                await self._set_status(paper_id, job_id, "processing", progress)

            records: list[VectorRecord] = []
            database_chunks: list[Chunk] = []
            title = parsed.title or original_title
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk_id = new_id()
                database_chunks.append(
                    Chunk(
                        id=chunk_id,
                        paper_id=paper_id,
                        page=chunk.page,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        token_count=chunk.token_count,
                        vector_id=chunk_id,
                    )
                )
                records.append(
                    VectorRecord(
                        id=chunk_id,
                        vector=vector,
                        payload={
                            "chunk_id": chunk_id,
                            "paper_id": paper_id,
                            "paper_title": title,
                            "page": chunk.page,
                            "text": chunk.text,
                        },
                    )
                )

            await self.vector_store.upsert(records)
            async with self.session_factory() as session:
                await session.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
                session.add_all(database_chunks)
                paper = await session.get(Paper, paper_id)
                job = await session.get(IngestionJob, job_id)
                if paper:
                    paper.title = title
                    paper.page_count = len(parsed.pages)
                    paper.status = "ready"
                    paper.error_message = None
                if job:
                    job.status = "completed"
                    job.progress = 100
                    job.error_message = None
                await session.commit()
        except Exception as exc:
            logger.exception("Paper ingestion failed for %s", paper_id)
            message = exc.message if isinstance(exc, AppError) else "论文处理失败，请查看服务日志。"
            await self._set_status(paper_id, job_id, "failed", 100, message)

    async def _set_status(
        self,
        paper_id: str,
        job_id: str,
        status: str,
        progress: int,
        error: str | None = None,
    ) -> None:
        async with self.session_factory() as session:
            paper = await session.get(Paper, paper_id)
            job = await session.get(IngestionJob, job_id)
            if paper:
                paper.status = "failed" if status == "failed" else status
                paper.error_message = error
            if job:
                job.status = status
                job.progress = progress
                job.error_message = error
            await session.commit()
