import asyncio
import hashlib
import logging
from collections import defaultdict
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import AppError
from app.db.models import Chunk, IngestionJob, Paper
from app.rag.index_profile import IndexProfile
from app.rag.pdf import ParsedPdf, parse_pdf
from app.rag.vector_store import LlamaIndexVectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: LlamaIndexVectorStore,
        profile: IndexProfile,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.profile = profile
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def process(self, paper_id: str, job_id: str) -> None:
        job_type = "ingest"
        try:
            async with self.session_factory() as session:
                paper = await session.get(Paper, paper_id)
                job = await session.get(IngestionJob, job_id)
                if paper is None or job is None:
                    raise AppError("论文或摄取任务不存在。", status_code=404, code="job_not_found")
                job_type = job.job_type
                file_path = self.settings.upload_dir / paper.stored_filename
                original_title = paper.title

            await self._set_status(paper_id, job_id, job_type, "processing", 5)
            parsed = await asyncio.to_thread(
                parse_pdf,
                file_path,
                self.settings.max_pdf_pages,
                table_ocr_enabled=self.settings.table_ocr_enabled,
                table_ocr_dpi=self.settings.table_ocr_dpi,
                table_ocr_min_confidence=self.settings.table_ocr_min_confidence,
            )
            title = parsed.title or original_title
            nodes = await asyncio.to_thread(self._build_nodes, parsed, paper_id, title)
            if not nodes:
                raise AppError("论文没有生成可索引文本。", code="empty_index")
            await self._set_status(paper_id, job_id, job_type, "processing", 25)

            await self.vector_store.add_nodes(nodes)
            await self._set_status(paper_id, job_id, job_type, "processing", 85)

            database_chunks = [
                Chunk(
                    id=node.node_id,
                    paper_id=paper_id,
                    page=int(node.metadata["page"]),
                    chunk_index=int(node.metadata["chunk_index"]),
                    text=node.get_content(),
                    token_count=len(self._encoding.encode(node.get_content())),
                    vector_id=node.node_id,
                    index_profile=self.profile.profile_id,
                )
                for node in nodes
            ]
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
                    paper.index_status = "ready"
                    paper.index_profile = self.profile.profile_id
                    paper.indexed_at = datetime.now(UTC)
                if job:
                    job.status = "completed"
                    job.progress = 100
                    job.error_message = None
                    job.details = {
                        "profile_id": self.profile.profile_id,
                        "provider": self.profile.provider,
                        "model": self.profile.model,
                        "dimensions": self.profile.dimensions,
                        "nodes": len(nodes),
                        "table_ocr_pages": parsed.table_ocr_pages,
                    }
                await session.commit()
        except Exception as exc:
            logger.exception("Paper ingestion failed for %s", paper_id)
            message = exc.message if isinstance(exc, AppError) else "论文处理失败，请查看服务日志。"
            await self._set_status(paper_id, job_id, job_type, "failed", 100, message)

    def _build_nodes(self, parsed: ParsedPdf, paper_id: str, title: str) -> list[BaseNode]:
        documents = [
            Document(
                id_=f"{paper_id}:page:{page.page}",
                text=page.text,
                metadata={
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page": page.page,
                    "index_profile": self.profile.profile_id,
                },
            )
            for page in parsed.pages
            if page.text.strip()
        ]
        splitter = SentenceSplitter(
            chunk_size=self.settings.chunk_size_tokens,
            chunk_overlap=self.settings.chunk_overlap_tokens,
            include_metadata=True,
            include_prev_next_rel=False,
        )
        nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
        per_page: dict[int, int] = defaultdict(int)
        for node in nodes:
            page = int(node.metadata["page"])
            chunk_index = per_page[page]
            per_page[page] += 1
            text_hash = hashlib.sha256(node.get_content().encode("utf-8")).hexdigest()
            node_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"researchflow:{paper_id}:{page}:{chunk_index}:{text_hash}",
                )
            )
            node.id_ = node_id
            node.metadata.update(
                {
                    "chunk_id": node_id,
                    "chunk_index": chunk_index,
                    "index_profile": self.profile.profile_id,
                }
            )
            metadata_keys = list(node.metadata)
            node.excluded_embed_metadata_keys = metadata_keys
            node.excluded_llm_metadata_keys = metadata_keys
        return nodes

    async def _set_status(
        self,
        paper_id: str,
        job_id: str,
        job_type: str,
        status: str,
        progress: int,
        error: str | None = None,
    ) -> None:
        async with self.session_factory() as session:
            paper = await session.get(Paper, paper_id)
            job = await session.get(IngestionJob, job_id)
            if paper:
                paper.index_status = "indexing" if status == "processing" else status
                paper.error_message = error
                if job_type == "ingest":
                    paper.status = "failed" if status == "failed" else status
            if job:
                job.status = status
                job.progress = progress
                job.error_message = error
            await session.commit()