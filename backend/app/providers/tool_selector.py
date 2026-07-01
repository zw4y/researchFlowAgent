import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

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


class OpenAIToolSelector:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        thinking: Literal["default", "enabled", "disabled"] = "default",
    ) -> None:
        self.model = model
        self._extra_body = None if thinking == "default" else {"thinking": {"type": thinking}}
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
            extra_body=self._extra_body,
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
