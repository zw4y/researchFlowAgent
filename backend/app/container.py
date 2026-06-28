from typing import cast

from pydantic import BaseModel

from app.agent.workflow import ResearchWorkflow
from app.core.config import Settings
from app.db.session import Database
from app.providers.base import ChatProvider, EmbeddingProvider
from app.providers.fake import FakeChatProvider, FakeEmbeddingProvider
from app.providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.providers.search import TavilySearchProvider
from app.providers.tool_selector import FakeToolSelector, OpenAIToolSelector, ToolSelector
from app.rag.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore
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
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)
        if settings.llm_mode == "fake":
            self.chat_provider: ChatProvider = FakeChatProvider()
            self.embedding_provider: EmbeddingProvider = FakeEmbeddingProvider()
            self.tool_selector: ToolSelector = FakeToolSelector()
        else:
            self.chat_provider = OpenAICompatibleChatProvider(
                settings.chat_api_key,
                settings.chat_base_url,
                settings.chat_model,
            )
            self.embedding_provider = OpenAICompatibleEmbeddingProvider(
                settings.embedding_api_key,
                settings.embedding_base_url,
                settings.embedding_model,
            )
            self.tool_selector = OpenAIToolSelector(
                settings.chat_api_key,
                settings.chat_base_url,
                settings.chat_model,
            )
        if settings.vector_store_mode == "qdrant":
            self.vector_store: VectorStore = QdrantVectorStore(
                settings.qdrant_url,
                settings.qdrant_collection,
            )
        else:
            self.vector_store = InMemoryVectorStore()
        self.search_provider = TavilySearchProvider(settings.tavily_api_key)
        sessions = self.database.session_factory
        self.paper_service = PaperService(settings, sessions, self.vector_store)
        self.ingestion_service = IngestionService(
            settings,
            sessions,
            self.embedding_provider,
            self.vector_store,
        )
        self.retrieval_service = RetrievalService(
            settings,
            sessions,
            self.embedding_provider,
            self.vector_store,
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
        await self.retrieval_service.rehydrate_memory_store()

    async def close(self) -> None:
        await self.vector_store.close()
        await self.database.dispose()
