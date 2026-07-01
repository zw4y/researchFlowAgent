from dataclasses import dataclass
from typing import Any, Literal, Protocol

RouteName = Literal["rag", "web", "metrics", "direct"]
RetrievalStatus = Literal["vector", "reranked", "rerank_fallback"]


@dataclass(slots=True)
class Evidence:
    chunk_id: str
    paper_id: str
    paper_title: str
    page: int
    text: str
    score: float
    retrieval_status: RetrievalStatus = "vector"
    vector_score: float | None = None


@dataclass(slots=True)
class WebResult:
    title: str
    url: str
    content: str
    score: float | None = None


@dataclass(slots=True)
class RerankResult:
    index: int
    score: float


class ChatProvider(Protocol):
    name: str

    async def chat(self, messages: list[dict[str, str]]) -> str: ...

    async def structured(self, prompt: str, schema_name: str) -> dict[str, Any]: ...

    async def route(
        self, question: str, *, has_papers: bool, enable_web: bool
    ) -> list[RouteName]: ...

    async def synthesize(
        self,
        question: str,
        evidence: list[Evidence],
        web_results: list[WebResult],
        metric_rows: list[dict[str, Any]],
    ) -> str: ...

    async def generate_metric_sql(self, question: str, paper_ids: list[str]) -> str: ...


class EmbeddingProvider(Protocol):
    name: str
    model_name: str
    dimensions: int
    profile_id: str
    configured: bool

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


class RerankProvider(Protocol):
    name: str
    model_name: str
    enabled: bool
    configured: bool

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]: ...


class SearchProvider(Protocol):
    name: str
    enabled: bool

    async def search(self, query: str, max_results: int = 5) -> list[WebResult]: ...