import json
from collections.abc import AsyncIterator
from typing import Literal, cast

from fastapi import APIRouter, BackgroundTasks, Depends, File, Response, UploadFile, status
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.container import AppContainer
from app.core.errors import AppError
from app.db.models import AgentRun, Conversation, Message, ToolCall
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    HealthResponse,
    IndexStatusResponse,
    IngestionJobOut,
    MessageOut,
    MetricImportResponse,
    PaperOut,
    PaperUploadResponse,
    ReindexResponse,
    StructuredChatRequest,
    StructuredChatResponse,
    ToolCallOut,
)

from .dependencies import get_container

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(container: AppContainer = Depends(get_container)) -> HealthResponse:
    llm_status = container.chat_provider.name
    if not container.settings.chat_api_key:
        llm_status = "not_configured"
    embedding_status = (
        container.embedding_provider.name
        if container.embedding_provider.configured
        else "not_configured"
    )
    rerank_status = container.rerank_provider.name
    if container.rerank_provider.enabled and not container.rerank_provider.configured:
        rerank_status = "not_configured"
    degraded = "not_configured" in {llm_status, embedding_status, rerank_status}
    return HealthResponse(
        status="degraded" if degraded else "ok",
        database=container.database.engine.dialect.name,
        vector_store=container.vector_store.name,
        llm=llm_status,
        embedding=embedding_status,
        rerank=rerank_status,
        index_profile=container.index_profile.profile_id,
        web_search="enabled" if container.search_provider.enabled else "disabled",
    )


@router.get("/index/status", response_model=IndexStatusResponse, tags=["system"])
async def index_status(
    container: AppContainer = Depends(get_container),
) -> IndexStatusResponse:
    return IndexStatusResponse(
        provider=container.embedding_provider.name,
        model=container.embedding_provider.model_name,
        dimensions=container.embedding_provider.dimensions,
        profile_id=container.index_profile.profile_id,
        collection=container.index_profile.collection_name,
        vector_store_mode=container.vector_store.name,
        collection_ready=await container.vector_store.current_collection_exists(),
        point_count=await container.vector_store.current_point_count(),
        paper_counts=await container.paper_service.index_status_counts(),
        embedding_configured=container.embedding_provider.configured,
        rerank_provider=container.rerank_provider.name,
        rerank_model=container.rerank_provider.model_name,
        rerank_configured=container.rerank_provider.configured,
    )


@router.get("/tools", tags=["system"])
async def list_tools(container: AppContainer = Depends(get_container)) -> list[dict]:
    return container.tools.schemas()


@router.post(
    "/papers",
    response_model=PaperUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["papers"],
)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
) -> PaperUploadResponse:
    if not container.embedding_provider.configured:
        raise AppError(
            "Embedding 服务尚未配置，无法上传并索引论文。",
            status_code=503,
            code="embedding_not_configured",
        )
    paper, job, duplicated = await container.paper_service.create_upload(file)
    if not duplicated or paper.status in {"pending", "failed"}:
        background_tasks.add_task(container.ingestion_service.process, paper.id, job.id)
    return PaperUploadResponse(paper=paper, ingestion_job=job, duplicated=duplicated)


@router.get("/papers", response_model=list[PaperOut], tags=["papers"])
async def list_papers(
    container: AppContainer = Depends(get_container),
) -> list[PaperOut]:
    return [PaperOut.model_validate(item) for item in await container.paper_service.list_papers()]


@router.get("/papers/{paper_id}", response_model=PaperOut, tags=["papers"])
async def get_paper(paper_id: str, container: AppContainer = Depends(get_container)) -> PaperOut:
    return PaperOut.model_validate(await container.paper_service.get_paper(paper_id))


@router.post(
    "/papers/{paper_id}/reindex",
    response_model=ReindexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["papers"],
)
async def reindex_paper(
    paper_id: str,
    background_tasks: BackgroundTasks,
    container: AppContainer = Depends(get_container),
) -> ReindexResponse:
    if not container.embedding_provider.configured:
        raise AppError(
            "Embedding 服务尚未配置，无法重建索引。",
            status_code=503,
            code="embedding_not_configured",
        )
    paper, job = await container.paper_service.create_reindex_job(paper_id)
    background_tasks.add_task(container.ingestion_service.process, paper.id, job.id)
    return ReindexResponse(paper_id=paper.id, job_id=job.id)


@router.delete(
    "/papers/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["papers"],
)
async def delete_paper(paper_id: str, container: AppContainer = Depends(get_container)) -> Response:
    await container.paper_service.delete_paper(paper_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/ingestion-jobs/{job_id}",
    response_model=IngestionJobOut,
    tags=["papers"],
)
async def get_ingestion_job(
    job_id: str, container: AppContainer = Depends(get_container)
) -> IngestionJobOut:
    return IngestionJobOut.model_validate(await container.paper_service.get_job(job_id))


@router.post(
    "/papers/{paper_id}/metrics/import",
    response_model=MetricImportResponse,
    tags=["metrics"],
)
async def import_metrics(
    paper_id: str,
    file: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
) -> MetricImportResponse:
    if not (file.filename or "").lower().endswith(".csv"):
        raise AppError("实验指标必须使用 CSV 文件。", code="unsupported_file_type")
    count = await container.metric_service.import_csv(paper_id, await file.read())
    return MetricImportResponse(paper_id=paper_id, imported=count)


@router.post("/llm/structured", response_model=StructuredChatResponse, tags=["llm"])
async def structured_chat(
    request: StructuredChatRequest,
    container: AppContainer = Depends(get_container),
) -> StructuredChatResponse:
    data = await container.chat_provider.structured(request.prompt, request.schema_name)
    return StructuredChatResponse(data=data)


@router.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(
    request: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    return await container.workflow.run(request)


@router.post("/chat/stream", tags=["agent"])
async def stream_chat(
    request: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> EventSourceResponse:
    async def events() -> AsyncIterator[dict[str, str]]:
        yield {"event": "run.started", "data": json.dumps({"question": request.question})}
        try:
            response = await container.workflow.run(request)
            for route_name in response.routes:
                yield {
                    "event": "node.completed",
                    "data": json.dumps({"node": route_name}),
                }
            for tool_call in response.tool_calls:
                yield {
                    "event": "tool.completed",
                    "data": tool_call.model_dump_json(),
                }
            for citation in response.citations:
                yield {"event": "citation", "data": citation.model_dump_json()}
            for start in range(0, len(response.answer), 36):
                yield {
                    "event": "token",
                    "data": json.dumps({"text": response.answer[start : start + 36]}),
                }
            yield {"event": "run.completed", "data": response.model_dump_json()}
        except AppError as exc:
            yield {
                "event": "run.failed",
                "data": json.dumps({"code": exc.code, "message": exc.message}),
            }
        except Exception:
            yield {
                "event": "run.failed",
                "data": json.dumps({"code": "internal_error", "message": "研究工作流执行失败。"}),
            }

    return EventSourceResponse(events())


@router.get("/conversations", tags=["conversations"])
async def list_conversations(
    container: AppContainer = Depends(get_container),
) -> list[dict]:
    async with container.database.session_factory() as session:
        result = await session.scalars(
            select(Conversation).order_by(Conversation.updated_at.desc()).limit(50)
        )
        return [
            {
                "id": item.id,
                "title": item.title,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in result
        ]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationOut,
    tags=["conversations"],
)
async def get_conversation(
    conversation_id: str,
    container: AppContainer = Depends(get_container),
) -> ConversationOut:
    async with container.database.session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            raise AppError("会话不存在。", status_code=404, code="conversation_not_found")
        messages = list(
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            )
        )
        run_ids = list(
            await session.scalars(
                select(AgentRun.id).where(AgentRun.conversation_id == conversation_id)
            )
        )
        calls: list[ToolCall] = []
        if run_ids:
            calls = list(
                await session.scalars(
                    select(ToolCall)
                    .where(ToolCall.run_id.in_(run_ids))
                    .order_by(ToolCall.created_at)
                )
            )
    return ConversationOut(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageOut.model_validate(item) for item in messages],
        tool_calls=[
            ToolCallOut(
                id=item.id,
                name=item.name,
                arguments=item.arguments,
                status=cast(Literal["completed", "failed"], item.status),
                duration_ms=item.duration_ms,
                result_summary=item.result_summary,
                error_message=item.error_message,
            )
            for item in calls
        ],
    )
