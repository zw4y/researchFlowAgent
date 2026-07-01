import hashlib
import math
import re
from typing import Any

from app.providers.base import Evidence, RerankResult, RouteName, WebResult
from app.providers.tool_selector import PlannedToolCall


class DeterministicChatProvider:
    name = "test-double"

    async def chat(self, messages: list[dict[str, str]]) -> str:
        question = messages[-1]["content"] if messages else ""
        return f"Deterministic test response: {question}"

    async def structured(self, prompt: str, schema_name: str) -> dict[str, Any]:
        if schema_name == "research_plan":
            return {
                "goal": prompt[:120],
                "steps": ["retrieve", "verify", "synthesize"],
                "risks": ["insufficient evidence"],
            }
        return {
            "title": "Deterministic summary",
            "problem": prompt[:160],
            "method": "Offline test double",
            "findings": ["The structured-output path works."],
        }

    async def route(
        self, question: str, *, has_papers: bool, enable_web: bool
    ) -> list[RouteName]:
        lowered = question.lower()
        routes: list[RouteName] = []
        if has_papers:
            routes.append("rag")
        if any(term in lowered for term in ("指标", "metric", "accuracy")):
            routes.append("metrics")
        if enable_web and any(term in lowered for term in ("最新", "web", "current")):
            routes.append("web")
        return routes or ["direct"]

    async def synthesize(
        self,
        question: str,
        evidence: list[Evidence],
        web_results: list[WebResult],
        metric_rows: list[dict[str, Any]],
    ) -> str:
        del web_results, metric_rows
        if not evidence:
            return f"Deterministic test response: {question}"
        return "\n".join(
            f"{item.text[:180]} [{item.paper_title}, page {item.page}]" for item in evidence[:3]
        )

    async def generate_metric_sql(self, question: str, paper_ids: list[str]) -> str:
        del question
        if paper_ids:
            quoted_ids = ", ".join(f"'{paper_id}'" for paper_id in paper_ids)
            return (
                "SELECT paper_id, experiment, metric_name, metric_value, unit, split "
                f"FROM experiment_metrics WHERE paper_id IN ({quoted_ids}) LIMIT 50"
            )
        return (
            "SELECT paper_id, experiment, metric_name, metric_value, unit, split "
            "FROM experiment_metrics LIMIT 50"
        )


class DeterministicEmbeddingProvider:
    name = "test-double"
    model_name = "deterministic-hash-v1"
    dimensions = 64
    profile_id = "deterministic-hash-v1-64"
    configured = True

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            vector[index] += 1.0 if digest[2] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class DeterministicRerankProvider:
    name = "test-double"
    model_name = "deterministic-rerank-v1"
    enabled = True
    configured = True

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        del query
        return [
            RerankResult(index=index, score=max(0.0, 1.0 - index * 0.01))
            for index in range(min(top_n, len(documents)))
        ]


class DeterministicToolSelector:
    async def select(
        self, question: str, tool_schemas: list[dict[str, Any]]
    ) -> list[PlannedToolCall]:
        del tool_schemas
        lowered = question.lower()
        calls: list[PlannedToolCall] = []
        if any(term in lowered for term in ("最新", "web", "current")):
            calls.append(PlannedToolCall("web_search", {"query": question, "max_results": 5}))
        if any(term in lowered for term in ("指标", "metric", "accuracy")):
            calls.append(PlannedToolCall("query_metrics", {"sql": ""}))
        return calls
