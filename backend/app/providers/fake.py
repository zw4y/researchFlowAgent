import hashlib
import math
import re
from typing import Any

from app.providers.base import Evidence, RouteName, WebResult


class FakeChatProvider:
    name = "fake"

    async def chat(self, messages: list[dict[str, str]]) -> str:
        question = messages[-1]["content"] if messages else ""
        return f"这是本地 Fake LLM 的回复：{question}"

    async def structured(self, prompt: str, schema_name: str) -> dict[str, Any]:
        if schema_name == "research_plan":
            return {
                "goal": prompt[:120],
                "steps": ["检索相关论文", "核对实验指标", "综合证据并生成引用"],
                "risks": ["来源覆盖不足", "不同论文实验设置不可直接比较"],
            }
        return {
            "title": "Local structured summary",
            "problem": prompt[:160],
            "method": "由 Fake Provider 生成，用于验证结构化输出链路。",
            "findings": ["接口与 Pydantic 校验工作正常。"],
        }

    async def route(self, question: str, *, has_papers: bool, enable_web: bool) -> list[RouteName]:
        lowered = question.lower()
        routes: list[RouteName] = []
        metric_terms = ("指标", "准确率", "精度", "召回", "f1", "metric", "accuracy", "对比")
        web_terms = ("联网", "最新", "网页", "web", "news", "current", "近期")
        if has_papers:
            routes.append("rag")
        if any(term in lowered for term in metric_terms):
            routes.append("metrics")
        if enable_web and any(term in lowered for term in web_terms):
            routes.append("web")
        if not routes:
            routes.append("direct")
        return routes

    async def synthesize(
        self,
        question: str,
        evidence: list[Evidence],
        web_results: list[WebResult],
        metric_rows: list[dict[str, Any]],
    ) -> str:
        sections = [f"关于“{question}”，当前工作流得到以下结果："]
        if evidence:
            sections.append("\n论文证据：")
            for index, item in enumerate(evidence[:3], start=1):
                excerpt = re.sub(r"\s+", " ", item.text).strip()[:220]
                sections.append(f"{index}. {excerpt} [{item.paper_title}，第 {item.page} 页]")
        if metric_rows:
            sections.append("\n实验指标：")
            for row in metric_rows[:5]:
                sections.append(
                    f"- {row.get('experiment')}: {row.get('metric_name')} = "
                    f"{row.get('metric_value')} {row.get('unit') or ''}".rstrip()
                )
        if web_results:
            sections.append("\n联网结果：")
            for web_item in web_results[:3]:
                sections.append(f"- {web_item.title}: {web_item.content[:180]} [{web_item.url}]")
        if not evidence and not metric_rows and not web_results:
            sections.append("\n尚未获得可验证的外部证据，请上传论文或启用对应工具。")
        return "\n".join(sections)

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


class FakeEmbeddingProvider:
    name = "fake"
    dimensions = 64

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
