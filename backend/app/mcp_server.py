from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from sqlalchemy import select

from app.container import AppContainer
from app.core.config import get_settings
from app.db.models import Citation
from app.schemas import ChatRequest

_shared_container: AppContainer | None = None


@dataclass(slots=True)
class McpAppContext:
    container: AppContainer


@asynccontextmanager
async def mcp_lifespan(_: FastMCP) -> AsyncIterator[McpAppContext]:
    owns_container = _shared_container is None
    container = _shared_container or AppContainer(get_settings())
    if owns_container:
        await container.start()
    try:
        yield McpAppContext(container=container)
    finally:
        if owns_container:
            await container.close()


mcp = FastMCP(
    "ResearchFlow Agent",
    instructions="Traceable research tools for papers, citations, and experiment metrics.",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
    lifespan=mcp_lifespan,
)


def configure_mcp(container: AppContainer) -> None:
    global _shared_container
    _shared_container = container


def _container(ctx: Context) -> AppContainer:
    app_context: McpAppContext = ctx.request_context.lifespan_context
    return app_context.container


@mcp.tool()
async def search_papers(query: str, ctx: Context) -> list[dict[str, Any]]:
    """Search uploaded paper metadata by title or filename."""
    container = _container(ctx)
    papers = await container.paper_service.list_papers()
    lowered = query.lower()
    return [
        {
            "id": paper.id,
            "title": paper.title,
            "filename": paper.original_filename,
            "pages": paper.page_count,
            "status": paper.status,
        }
        for paper in papers
        if lowered in paper.title.lower() or lowered in paper.original_filename.lower()
    ][:20]


@mcp.tool()
async def ask_paper(
    question: str,
    paper_ids: list[str],
    ctx: Context,
) -> dict[str, Any]:
    """Ask a question about uploaded papers and receive traceable citations."""
    response = await _container(ctx).workflow.run(
        ChatRequest(question=question, paper_ids=paper_ids, enable_web=False)
    )
    return response.model_dump(mode="json")


@mcp.tool()
async def query_experiment_metrics(
    question: str,
    paper_ids: list[str],
    ctx: Context,
) -> list[dict[str, Any]]:
    """Answer a numerical experiment question through validated read-only SQL."""
    container = _container(ctx)
    sql = await container.chat_provider.generate_metric_sql(question, paper_ids)
    return await container.metric_service.execute_readonly(sql)


@mcp.tool()
async def get_citations(message_id: str, ctx: Context) -> list[dict[str, Any]]:
    """Return all citations stored for an assistant message."""
    container = _container(ctx)
    async with container.database.session_factory() as session:
        citations = list(
            await session.scalars(
                select(Citation)
                .where(Citation.message_id == message_id)
                .order_by(Citation.created_at)
            )
        )
    return [
        {
            "source_type": item.source_type,
            "paper_id": item.paper_id,
            "paper_title": item.paper_title,
            "page": item.page,
            "chunk_id": item.chunk_id,
            "url": item.url,
            "source_title": item.source_title,
            "excerpt": item.excerpt,
            "score": item.score,
        }
        for item in citations
    ]


def run_stdio() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
