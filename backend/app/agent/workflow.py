from datetime import UTC, datetime
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError
from app.db.models import AgentRun, Citation, Conversation, Message, Paper
from app.providers.base import ChatProvider, Evidence, WebResult
from app.providers.tool_selector import PlannedToolCall, ToolSelector
from app.schemas import ChatRequest, ChatResponse, CitationOut, ToolCallOut
from app.services.metrics import MetricService
from app.tools.registry import ToolRegistry


class AgentState(TypedDict, total=False):
    question: str
    paper_ids: list[str]
    enable_web: bool
    run_id: str
    planned_tools: list[dict[str, Any]]
    routes: list[str]
    evidence: list[Evidence]
    web_results: list[WebResult]
    metric_rows: list[dict[str, Any]]
    tool_calls: list[ToolCallOut]
    answer: str
    citations: list[CitationOut]
    grounding_status: str


class ResearchWorkflow:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        chat_provider: ChatProvider,
        tool_selector: ToolSelector,
        tools: ToolRegistry,
        metric_service: MetricService,
    ) -> None:
        self.session_factory = session_factory
        self.chat_provider = chat_provider
        self.tool_selector = tool_selector
        self.tools = tools
        self.metric_service = metric_service
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("intent_router", self._route)
        builder.add_node("rag_node", self._rag)
        builder.add_node("web_search_node", self._web)
        builder.add_node("sql_query_node", self._metrics)
        builder.add_node("answer_synthesis_node", self._synthesize)
        builder.add_node("citation_check_node", self._check_citations)
        builder.add_edge(START, "intent_router")
        builder.add_edge("intent_router", "rag_node")
        builder.add_edge("rag_node", "web_search_node")
        builder.add_edge("web_search_node", "sql_query_node")
        builder.add_edge("sql_query_node", "answer_synthesis_node")
        builder.add_edge("answer_synthesis_node", "citation_check_node")
        builder.add_edge("citation_check_node", END)
        return builder.compile()

    async def _route(self, state: AgentState) -> dict[str, Any]:
        schemas = self.tools.schemas()
        if not state["enable_web"]:
            schemas = [item for item in schemas if item["function"]["name"] != "web_search"]
        planned = await self.tool_selector.select(state["question"], schemas)
        names = {item.name for item in planned}
        if state["paper_ids"] and not {"search_documents", "ask_paper"} & names:
            planned.insert(
                0,
                PlannedToolCall(
                    "search_documents",
                    {"query": state["question"], "paper_ids": state["paper_ids"]},
                ),
            )
            names.add("search_documents")
        routes: list[str] = []
        if {"search_documents", "ask_paper"} & names:
            routes.append("rag")
        if "web_search" in names:
            routes.append("web")
        if "query_metrics" in names:
            routes.append("metrics")
        if not routes:
            routes.append("direct")
        return {
            "planned_tools": [{"name": item.name, "arguments": item.arguments} for item in planned],
            "routes": routes,
            "evidence": [],
            "web_results": [],
            "metric_rows": [],
            "tool_calls": [],
        }

    async def _rag(self, state: AgentState) -> dict[str, Any]:
        if "rag" not in state["routes"]:
            return {}
        arguments = {
            "query": state["question"],
            "paper_ids": state["paper_ids"],
        }
        execution = await self.tools.execute("search_documents", arguments, state["run_id"])
        return {
            "evidence": execution.value or [],
            "tool_calls": [*state["tool_calls"], execution.trace],
        }

    async def _web(self, state: AgentState) -> dict[str, Any]:
        if "web" not in state["routes"] or not state["enable_web"]:
            return {}
        execution = await self.tools.execute(
            "web_search",
            {"query": state["question"], "max_results": 5},
            state["run_id"],
        )
        return {
            "web_results": execution.value or [],
            "tool_calls": [*state["tool_calls"], execution.trace],
        }

    async def _metrics(self, state: AgentState) -> dict[str, Any]:
        if "metrics" not in state["routes"]:
            return {}
        sql = await self.chat_provider.generate_metric_sql(state["question"], state["paper_ids"])
        execution = await self.tools.execute("query_metrics", {"sql": sql}, state["run_id"])
        return {
            "metric_rows": execution.value or [],
            "tool_calls": [*state["tool_calls"], execution.trace],
        }

    async def _synthesize(self, state: AgentState) -> dict[str, Any]:
        answer = await self.chat_provider.synthesize(
            state["question"],
            state["evidence"],
            state["web_results"],
            state["metric_rows"],
        )
        return {"answer": answer}

    async def _check_citations(self, state: AgentState) -> dict[str, Any]:
        citations = [
            CitationOut(
                source_type="paper",
                paper_id=item.paper_id,
                paper_title=item.paper_title,
                page=item.page,
                chunk_id=item.chunk_id,
                excerpt=" ".join(item.text.split())[:500],
                score=round(item.score, 4),
            )
            for item in state["evidence"]
        ]
        citations.extend(
            CitationOut(
                source_type="web",
                url=item.url,
                source_title=item.title,
                excerpt=" ".join(item.content.split())[:500],
                score=item.score,
            )
            for item in state["web_results"]
            if item.url.startswith(("http://", "https://"))
        )
        failed = any(item.status == "failed" for item in state["tool_calls"])
        if citations and not failed:
            grounding_status = "grounded"
        elif citations or state["metric_rows"]:
            grounding_status = "partial"
        else:
            grounding_status = "unsupported"
        return {"citations": citations, "grounding_status": grounding_status}

    async def run(self, request: ChatRequest) -> ChatResponse:
        conversation = await self._get_or_create_conversation(request)
        async with self.session_factory() as session:
            user_message = Message(
                conversation_id=conversation.id,
                role="user",
                content=request.question,
            )
            run = AgentRun(
                conversation_id=conversation.id,
                question=request.question,
            )
            session.add_all([user_message, run])
            await session.commit()
            await session.refresh(run)

        initial: AgentState = {
            "question": request.question,
            "paper_ids": request.paper_ids,
            "enable_web": request.enable_web,
            "run_id": run.id,
        }
        try:
            state = await self.graph.ainvoke(
                initial,
                config={"configurable": {"thread_id": conversation.id}},
            )
        except Exception:
            async with self.session_factory() as session:
                failed_run = await session.get(AgentRun, run.id)
                if failed_run:
                    failed_run.status = "failed"
                    failed_run.completed_at = datetime.now(UTC)
                    await session.commit()
            raise

        async with self.session_factory() as session:
            assistant_message = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=state["answer"],
                message_metadata={
                    "routes": state["routes"],
                    "grounding_status": state["grounding_status"],
                    "run_id": run.id,
                },
            )
            session.add(assistant_message)
            await session.flush()
            for item in state["citations"]:
                session.add(
                    Citation(
                        message_id=assistant_message.id,
                        source_type=item.source_type,
                        paper_id=item.paper_id,
                        paper_title=item.paper_title,
                        page=item.page,
                        chunk_id=item.chunk_id,
                        url=item.url,
                        source_title=item.source_title,
                        excerpt=item.excerpt,
                        score=item.score,
                    )
                )
            completed_run = await session.get(AgentRun, run.id)
            if completed_run:
                completed_run.status = "completed"
                completed_run.routes = state["routes"]
                completed_run.grounding_status = state["grounding_status"]
                completed_run.completed_at = datetime.now(UTC)
            conversation_record = await session.get(Conversation, conversation.id)
            if conversation_record:
                conversation_record.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(assistant_message)

        return ChatResponse(
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            run_id=run.id,
            answer=state["answer"],
            citations=state["citations"],
            routes=state["routes"],
            tool_calls=state["tool_calls"],
            grounding_status=cast(
                Literal["grounded", "partial", "unsupported"],
                state["grounding_status"],
            ),
        )

    async def _get_or_create_conversation(self, request: ChatRequest) -> Conversation:
        async with self.session_factory() as session:
            if request.conversation_id:
                conversation = await session.get(Conversation, request.conversation_id)
                if conversation is None:
                    raise AppError("会话不存在。", status_code=404, code="conversation_not_found")
                return conversation
            conversation = Conversation(title=request.question[:80])
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation

    async def list_ready_paper_ids(self) -> list[str]:
        async with self.session_factory() as session:
            result = await session.scalars(select(Paper.id).where(Paper.status == "ready"))
            return list(result)
