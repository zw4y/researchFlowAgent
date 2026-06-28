import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam

from app.core.errors import AppError


@dataclass(slots=True)
class PlannedToolCall:
    name: str
    arguments: dict[str, Any]


class ToolSelector(Protocol):
    async def select(
        self, question: str, tool_schemas: list[dict[str, Any]]
    ) -> list[PlannedToolCall]: ...


class FakeToolSelector:
    async def select(
        self, question: str, tool_schemas: list[dict[str, Any]]
    ) -> list[PlannedToolCall]:
        del tool_schemas
        lowered = question.lower()
        calls: list[PlannedToolCall] = []
        if any(term in lowered for term in ("最新", "联网", "web", "current", "近期")):
            calls.append(PlannedToolCall("web_search", {"query": question, "max_results": 5}))
        if any(
            term in lowered
            for term in ("指标", "准确率", "精度", "召回", "f1", "metric", "accuracy", "对比")
        ):
            calls.append(PlannedToolCall("query_metrics", {"sql": ""}))
        return calls


class OpenAIToolSelector:
    def __init__(self, api_key: str | None, base_url: str, model: str) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url) if api_key else None

    async def select(
        self, question: str, tool_schemas: list[dict[str, Any]]
    ) -> list[PlannedToolCall]:
        if self.client is None:
            raise AppError(
                "未配置 CHAT_API_KEY，无法进行 Function Calling。",
                status_code=503,
                code="llm_not_configured",
            )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select all tools needed for this research question. "
                        "Use search_documents for uploaded papers, web_search for current facts, "
                        "and query_metrics for numerical experiment comparisons."
                    ),
                },
                {"role": "user", "content": question},
            ],
            tools=cast(Iterable[ChatCompletionToolParam], tool_schemas),
            tool_choice="auto",
            temperature=0,
        )
        calls: list[PlannedToolCall] = []
        for item in response.choices[0].message.tool_calls or []:
            if item.type != "function":
                continue
            try:
                arguments = json.loads(item.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            calls.append(PlannedToolCall(name=item.function.name, arguments=arguments))
        return calls
