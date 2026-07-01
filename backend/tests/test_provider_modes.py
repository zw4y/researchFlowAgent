from app.container import AppContainer
from app.core.config import Settings
from app.providers.openai_compatible import OpenAICompatibleChatProvider
from app.providers.search import TavilySearchProvider

from tests.doubles import (
    DeterministicChatProvider,
    DeterministicEmbeddingProvider,
    DeterministicRerankProvider,
    DeterministicToolSelector,
)


class StubChatProvider(OpenAICompatibleChatProvider):
    def __init__(self) -> None:
        super().__init__(None, "https://example.invalid", "test-model")
        self.received_messages: list[dict[str, str]] = []

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.received_messages = messages
        return "direct answer"


async def test_direct_synthesis_answers_without_evidence() -> None:
    provider = StubChatProvider()

    answer = await provider.synthesize("什么是 RAG？", [], [], [])

    assert answer == "direct answer"
    assert provider.received_messages[-1]["content"] == "什么是 RAG？"
    assert "Do not invent citations" in provider.received_messages[0]["content"]


async def test_container_accepts_offline_test_doubles() -> None:
    settings = Settings(
        chat_api_key="test-key",
        chat_base_url="https://example.invalid",
        chat_model="test-model",
        dashscope_api_key="test-key",
    )
    container = AppContainer(
        settings,
        chat_provider=DeterministicChatProvider(),
        tool_selector=DeterministicToolSelector(),
        embedding_provider=DeterministicEmbeddingProvider(),
        rerank_provider=DeterministicRerankProvider(),
    )
    try:
        assert isinstance(container.embedding_provider, DeterministicEmbeddingProvider)
        assert container.chat_provider.name == "test-double"
    finally:
        await container.close()


async def test_tavily_uses_bearer_authentication(monkeypatch) -> None:
    captured: dict = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "title": "Source",
                        "url": "https://example.com",
                        "content": "Evidence",
                        "score": 0.9,
                    }
                ]
            }

    async def mock_post(client, url: str, **kwargs):
        del client
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("app.providers.search.httpx.AsyncClient.post", mock_post)

    results = await TavilySearchProvider("test-key").search("latest RAG research")

    assert captured["headers"] == {"Authorization": "Bearer test-key"}
    assert "api_key" not in captured["json"]
    assert results[0].url == "https://example.com"
