import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import AppError
from app.db.models import Chunk, IngestionJob, Paper
from app.rag.index_profile import IndexProfile
from app.rag.vector_store import LlamaIndexVectorStore


class PaperService:
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

    async def create_upload(self, upload: UploadFile) -> tuple[Paper, IngestionJob, bool]:
        filename = Path(upload.filename or "paper.pdf").name
        if Path(filename).suffix.lower() != ".pdf":
            raise AppError("只支持 PDF 文件。", code="unsupported_file_type")
        size_limit = self.settings.max_pdf_size_mb * 1024 * 1024
        content = await upload.read(size_limit + 1)
        if len(content) > size_limit:
            raise AppError(
                f"文件超过 {self.settings.max_pdf_size_mb} MB 限制。",
                status_code=413,
                code="pdf_size_limit_exceeded",
            )
        if not content.startswith(b"%PDF"):
            raise AppError("文件内容不是有效 PDF。", code="invalid_pdf")
        digest = hashlib.sha256(content).hexdigest()

        async with self.session_factory() as session:
            existing = await session.scalar(select(Paper).where(Paper.sha256 == digest))
            if existing:
                job = await session.scalar(
                    select(IngestionJob)
                    .where(IngestionJob.paper_id == existing.id)
                    .order_by(IngestionJob.created_at.desc())
                )
                if job is None:
                    job = IngestionJob(
                        paper_id=existing.id,
                        status=existing.status,
                        progress=100,
                        job_type="ingest",
                    )
                    session.add(job)
                    await session.commit()
                    await session.refresh(job)
                return existing, job, True

            paper_id = str(uuid4())
            stored_filename = f"{paper_id}.pdf"
            paper = Paper(
                id=paper_id,
                title=Path(filename).stem,
                original_filename=filename,
                stored_filename=stored_filename,
                sha256=digest,
                index_status="pending",
            )
            job = IngestionJob(
                paper_id=paper_id,
                job_type="ingest",
                details={"target_profile": self.profile.profile_id},
            )
            session.add_all([paper, job])
            await session.commit()
            await session.refresh(paper)
            await session.refresh(job)

        target = self.settings.upload_dir / stored_filename
        try:
            await asyncio.to_thread(target.write_bytes, content)
        except Exception:
            async with self.session_factory() as session:
                failed_paper = await session.get(Paper, paper_id)
                failed_job = await session.get(IngestionJob, job.id)
                if failed_paper:
                    failed_paper.status = "failed"
                    failed_paper.index_status = "failed"
                    failed_paper.error_message = "文件保存失败"
                if failed_job:
                    failed_job.status = "failed"
                    failed_job.error_message = "文件保存失败"
                await session.commit()
            raise
        return paper, job, False

    async def create_reindex_job(self, paper_id: str) -> tuple[Paper, IngestionJob]:
        async with self.session_factory() as session:
            paper = await session.get(Paper, paper_id)
            if paper is None:
                raise AppError("论文不存在。", status_code=404, code="paper_not_found")
            if not (self.settings.upload_dir / paper.stored_filename).exists():
                raise AppError("论文源文件不存在，无法重建索引。", code="paper_file_missing")
            running = await session.scalar(
                select(IngestionJob).where(
                    IngestionJob.paper_id == paper_id,
                    IngestionJob.status.in_(["queued", "processing"]),
                )
            )
            if running:
                raise AppError("该论文已有索引任务正在运行。", status_code=409, code="job_running")
            job = IngestionJob(
                paper_id=paper_id,
                job_type="reindex",
                details={"target_profile": self.profile.profile_id},
            )
            paper.index_status = "pending"
            paper.error_message = None
            session.add(job)
            await session.commit()
            await session.refresh(paper)
            await session.refresh(job)
            return paper, job

    async def mark_stale_indexes(self) -> int:
        async with self.session_factory() as session:
            result = await session.execute(
                update(Paper)
                .where(
                    Paper.status == "ready",
                    or_(
                        Paper.index_profile.is_(None),
                        Paper.index_profile != self.profile.profile_id,
                    ),
                    Paper.index_status != "indexing",
                )
                .values(index_status="stale")
            )
            await session.commit()
            return int(getattr(result, "rowcount", 0) or 0)

    async def index_status_counts(self) -> dict[str, int]:
        async with self.session_factory() as session:
            rows = await session.execute(
                select(Paper.index_status, func.count(Paper.id)).group_by(Paper.index_status)
            )
            return {str(status): int(count) for status, count in rows}

    async def list_papers(self) -> list[Paper]:
        async with self.session_factory() as session:
            result = await session.scalars(select(Paper).order_by(Paper.created_at.desc()))
            return list(result)

    async def get_paper(self, paper_id: str) -> Paper:
        async with self.session_factory() as session:
            paper = await session.get(Paper, paper_id)
            if paper is None:
                raise AppError("论文不存在。", status_code=404, code="paper_not_found")
            return paper

    async def delete_paper(self, paper_id: str) -> None:
        paper = await self.get_paper(paper_id)
        await self.vector_store.delete_paper(paper_id)
        async with self.session_factory() as session:
            await session.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
            await session.execute(delete(IngestionJob).where(IngestionJob.paper_id == paper_id))
            target = await session.get(Paper, paper_id)
            if target:
                await session.delete(target)
            await session.commit()
        path = self.settings.upload_dir / paper.stored_filename
        if path.exists():
            await asyncio.to_thread(path.unlink)

    async def get_job(self, job_id: str) -> IngestionJob:
        async with self.session_factory() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                raise AppError("摄取任务不存在。", status_code=404, code="job_not_found")
            return job