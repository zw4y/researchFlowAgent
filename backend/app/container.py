from typing import cast

from pydantic import BaseModel

from app.agent.workflow import ResearchWorkflow
from app.core.config import Settings
from app.db.session import Database
from app.providers.base import ChatProvider, EmbeddingProvider, RerankProvider
from app.providers.dashscope import DashScopeEmbeddingProvider, DashScopeRerankProvider
from app.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.providers.search import TavilySearchProvider
from app.providers.tool_selector import OpenAIToolSelector, ToolSelector
from app.rag.index_profile import IndexProfile
from app.rag.vector_store import LlamaIndexVectorStore
from app.services.ingestion import IngestionService
from app.services.metrics import MetricService
from app.services.papers import PaperService
from app.services.retrieval import RetrievalService
from app.tools.registry import (
    AskPaperArgs,
    CalculatorArgs,
    ManagedFileReader,
    QueryMetricsArgs,
    ReadFileArgs,
    SafeCalculator,
    SearchDocumentsArgs,
    ToolDefinition,
    ToolRegistry,
    WebSearchArgs,
)


class AppContainer:
    def __init__(
        self,
        settings: Settings,
        *,
        chat_provider: ChatProvider | None = None,
        tool_selector: ToolSelector | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        self.settings = settings
        self.database = Database(settings)
        if (chat_provider is None) != (tool_selector is None):
            raise ValueError("chat_provider and tool_selector must be injected together")
        if chat_provider is None or tool_selector is None:
            self.chat_provider, self.tool_selector = self._build_chat_provider()
        else:
            self.chat_provider, self.tool_selector = chat_provider, tool_selector
        self.embedding_provider = embedding_provider or self._build_embedding_provider()
        self.rerank_provider = rerank_provider or self._build_rerank_provider()
        self.index_profile = IndexProfile.build(settings, self.embedding_provider)
        self.vector_store = LlamaIndexVectorStore(
            settings,
            self.index_profile,
            self.embedding_provider,
        )
        self.search_provider = TavilySearchProvider(settings.tavily_api_key)
        sessions = self.database.session_factory
        self.paper_service = PaperService(
            settings,
            sessions,
            self.vector_store,
            self.index_profile,
        )
        self.ingestion_service = IngestionService(
            settings,
            sessions,
            self.vector_store,
            self.index_profile,
        )
        self.retrieval_service = RetrievalService(
            settings,
            sessions,
            self.vector_store,
            self.rerank_provider,
            self.index_profile,
        )
        self.metric_service = MetricService(sessions)
        self.tools = ToolRegistry(sessions)
        self._register_tools()
        self.workflow = ResearchWorkflow(
            sessions,
            self.chat_provider,
            self.tool_selector,
            self.tools,
            self.metric_service,
        )

    def _build_chat_provider(self) -> tuple[ChatProvider, ToolSelector]:
        return (
            OpenAICompatibleChatProvider(
                self.settings.chat_api_key,
                self.settings.chat_base_url,
                self.settings.chat_model,
                self.settings.chat_thinking,
            ),
            OpenAIToolSelector(
                self.settings.chat_api_key,
                self.settings.chat_base_url,
                self.settings.chat_model,
                self.settings.chat_thinking,
            ),
        )

    def _build_embedding_provider(self) -> EmbeddingProvider:
        if self.settings.embedding_mode == "dashscope":
            return DashScopeEmbeddingProvider(
                self.settings.dashscope_api_key,
                self.settings.embedding_model,
                self.settings.embedding_dimensions,
                self.settings.embedding_batch_size,
                self.settings.embedding_timeout_seconds,
                self.settings.embedding_max_retries,
                self.settings.embedding_query_instruction,
            )
        return OpenAICompatibleEmbeddingProvider(
            self.settings.embedding_api_key,
            self.settings.embedding_base_url,
            self.settings.embedding_model,
            self.settings.embedding_dimensions,
        )

    def _build_rerank_provider(self) -> RerankProvider:
        return DashScopeRerankProvider(
            self.settings.dashscope_api_key,
            self.settings.rerank_model,
            self.settings.rerank_timeout_seconds,
            self.settings.rerank_max_retries,
            self.settings.rerank_instruction,
        )

    def _register_tools(self) -> None:
        calculator = SafeCalculator()
        reader = ManagedFileReader(self.settings.upload_dir)

        async def search_documents(args: BaseModel):
            values = cast(SearchDocumentsArgs, args)
            return await self.retrieval_service.search(values.query, values.paper_ids)

        async def ask_paper(args: BaseModel):
            values = cast(AskPaperArgs, args)
            evidence = await self.retrieval_service.search(values.query, values.paper_ids)
            return await self.chat_provider.synthesize(values.query, evidence, [], [])

        async def web_search(args: BaseModel):
            values = cast(WebSearchArgs, args)
            return await self.search_provider.search(values.query, values.max_results)

        async def query_metrics(args: BaseModel):
            values = cast(QueryMetricsArgs, args)
            return await self.metric_service.execute_readonly(values.sql)

        async def calculate(args: BaseModel):
            values = cast(CalculatorArgs, args)
            return calculator.evaluate(values.expression)

        async def read_file(args: BaseModel):
            values = cast(ReadFileArgs, args)
            return await reader.read(values.relative_path)

        definitions = [
            ToolDefinition(
                "search_documents",
                "Search uploaded papers and return page-level evidence.",
                SearchDocumentsArgs,
                search_documents,
            ),
            ToolDefinition(
                "ask_paper",
                "Answer a question using one or more uploaded papers.",
                AskPaperArgs,
                ask_paper,
            ),
            ToolDefinition(
                "web_search",
                "Search the public web for current technical information.",
                WebSearchArgs,
                web_search,
            ),
            ToolDefinition(
                "query_metrics",
                "Execute a validated read-only SQL query against experiment metrics.",
                QueryMetricsArgs,
                query_metrics,
            ),
            ToolDefinition(
                "calculator",
                "Evaluate a basic arithmetic expression.",
                CalculatorArgs,
                calculate,
            ),
            ToolDefinition(
                "read_file",
                "Read a UTF-8 text, Markdown, or CSV file inside the managed upload directory.",
                ReadFileArgs,
                read_file,
            ),
        ]
        for definition in definitions:
            self.tools.register(definition)

    async def start(self) -> None:
        self.settings.prepare_directories()
        if self.settings.auto_create_tables:
            await self.database.create_tables()
        await self.paper_service.mark_stale_indexes()

    async def close(self) -> None:
        await self.vector_store.close()
        await self.database.dispose()