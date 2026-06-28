from dataclasses import dataclass
from typing import Any, Literal, Protocol

RouteName = Literal["rag", "web", "metrics", "direct"]


@dataclass(slots=True)
class Evidence:
    chunk_id: str
    paper_id: str
    paper_title: str
    page: int
    text: str
    score: float


@dataclass(slots=True)
class WebResult:
    title: str
    url: str
    content: str
    score: float | None = None


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

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SearchProvider(Protocol):
    name: str
    enabled: bool

    async def search(self, query: str, max_results: int = 5) -> list[WebResult]: ...
