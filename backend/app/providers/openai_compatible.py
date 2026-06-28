import json
from typing import Any, cast

from openai import AsyncOpenAI

from app.core.errors import AppError
from app.providers.base import Evidence, RouteName, WebResult


class OpenAICompatibleChatProvider:
    name = "openai_compatible"

    def __init__(self, api_key: str | None, base_url: str, model: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url) if api_key else None

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise AppError(
                "未配置 CHAT_API_KEY，无法调用真实模型。",
                status_code=503,
                code="llm_not_configured",
            )
        return self._client

    async def chat(self, messages: list[dict[str, str]]) -> str:
        response = await self._require_client().chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    async def structured(self, prompt: str, schema_name: str) -> dict[str, Any]:
        response = await self._require_client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Return one valid JSON object for schema '{schema_name}'. "
                        "Do not include markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content or "{}")

    async def route(self, question: str, *, has_papers: bool, enable_web: bool) -> list[RouteName]:
        prompt = (
            "Classify this research question into one or more routes. "
            "Allowed routes: rag, web, metrics, direct. "
            f"has_papers={has_papers}, enable_web={enable_web}. "
            "Use rag for uploaded papers, metrics for structured experiment values, "
            "web for current facts, and direct only when no tool is needed. "
            f'Question: {question}\nReturn JSON: {{"routes": ["rag"]}}'
        )
        data = await self.structured(prompt, "route_decision")
        allowed = {"rag", "web", "metrics", "direct"}
        routes = [item for item in data.get("routes", []) if item in allowed]
        if not enable_web:
            routes = [item for item in routes if item != "web"]
        if not has_papers:
            routes = [item for item in routes if item != "rag"]
        return cast(list[RouteName], routes or ["direct"])

    async def synthesize(
        self,
        question: str,
        evidence: list[Evidence],
        web_results: list[WebResult],
        metric_rows: list[dict[str, Any]],
    ) -> str:
        paper_context = "\n".join(
            f"[P{index}] {item.paper_title}, page {item.page}, chunk {item.chunk_id}: {item.text}"
            for index, item in enumerate(evidence, start=1)
        )
        web_context = "\n".join(
            f"[W{index}] {item.title} ({item.url}): {item.content}"
            for index, item in enumerate(web_results, start=1)
        )
        prompt = (
            "You are ResearchFlow, a careful research assistant. Answer in the language of "
            "the question. Use only the evidence below for factual claims. Cite paper evidence "
            "as [P1] and web evidence as [W1]. State uncertainty when evidence is insufficient.\n\n"
            f"Question:\n{question}\n\nPaper evidence:\n{paper_context or '(none)'}\n\n"
            f"Web evidence:\n{web_context or '(none)'}\n\nMetrics:\n"
            f"{json.dumps(metric_rows, ensure_ascii=False) if metric_rows else '(none)'}"
        )
        return await self.chat(
            [
                {"role": "system", "content": "Provide a concise, evidence-grounded answer."},
                {"role": "user", "content": prompt},
            ]
        )

    async def generate_metric_sql(self, question: str, paper_ids: list[str]) -> str:
        prompt = (
            "Generate one PostgreSQL SELECT statement. Only table experiment_metrics is "
            "allowed. Allowed columns: paper_id, experiment, metric_name, metric_value, unit, "
            "split, notes. No comments, no semicolon, max 50 rows. "
            f"Restrict paper_id to this list when non-empty: {paper_ids}. Question: {question}. "
            'Return JSON {"sql": "SELECT ..."}.'
        )
        data = await self.structured(prompt, "metric_sql")
        return str(data.get("sql", ""))


class OpenAICompatibleEmbeddingProvider:
    name = "openai_compatible"

    def __init__(self, api_key: str | None, base_url: str, model: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url) if api_key else None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise AppError(
                "未配置 EMBEDDING_API_KEY，无法生成向量。",
                status_code=503,
                code="embedding_not_configured",
            )
        response = await self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
