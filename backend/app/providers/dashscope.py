import asyncio
import hashlib
import logging
from collections.abc import Callable
from typing import Any

from dashscope import TextEmbedding, TextReRank  # type: ignore[import-untyped]

from app.core.errors import AppError
from app.providers.base import RerankResult

logger = logging.getLogger(__name__)
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class DashScopeEmbeddingProvider:
    name = "dashscope"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        dimensions: int,
        batch_size: int,
        timeout_seconds: int,
        max_retries: int,
        query_instruction: str,
    ) -> None:
        self.api_key = api_key
        self.model_name = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.query_instruction = query_instruction
        self.configured = bool(api_key)
        identity = f"{self.name}:{model}:{dimensions}"
        self.profile_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                await self._embed_batch(texts[start : start + self.batch_size], "document")
            )
        self._validate(vectors, len(texts))
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self._embed_batch([query], "query")
        self._validate(vectors, 1)
        return vectors[0]

    async def _embed_batch(self, texts: list[str], text_type: str) -> list[list[float]]:
        self._require_key()

        def request() -> Any:
            return TextEmbedding.call(
                model=self.model_name,
                input=texts,
                api_key=self.api_key,
                text_type=text_type,
                dimension=self.dimensions,
                output_type="dense",
                instruct=self.query_instruction if text_type == "query" else None,
                timeout=self.timeout_seconds,
            )

        response = await _call_with_retry(
            request,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            service_name="Embedding",
        )
        output = response.output or {}
        items = sorted(output.get("embeddings", []), key=lambda item: item.get("text_index", 0))
        return [list(item.get("embedding", [])) for item in items]

    def _require_key(self) -> None:
        if not self.api_key:
            raise AppError(
                "未配置 DASHSCOPE_API_KEY，无法生成论文向量。",
                status_code=503,
                code="embedding_not_configured",
            )

    def _validate(self, vectors: list[list[float]], expected: int) -> None:
        if len(vectors) != expected or any(len(item) != self.dimensions for item in vectors):
            raise AppError(
                "百炼 Embedding 返回的向量数量或维度不正确。",
                status_code=502,
                code="embedding_invalid_response",
            )


class DashScopeRerankProvider:
    name = "dashscope"
    enabled = True

    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout_seconds: int,
        max_retries: int,
        instruction: str,
    ) -> None:
        self.api_key = api_key
        self.model_name = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.instruction = instruction
        self.configured = bool(api_key)

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        if not documents:
            return []
        if not self.api_key:
            raise AppError(
                "未配置 DASHSCOPE_API_KEY，无法执行重排序。",
                status_code=503,
                code="rerank_not_configured",
            )

        def request() -> Any:
            return TextReRank.call(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=False,
                instruct=self.instruction,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )

        response = await _call_with_retry(
            request,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            service_name="Rerank",
        )
        results = getattr(response.output, "results", [])
        parsed = [
            RerankResult(index=int(item.index), score=float(item.relevance_score))
            for item in results
            if 0 <= int(item.index) < len(documents)
        ]
        if not parsed:
            raise AppError(
                "百炼 Rerank 未返回有效结果。",
                status_code=502,
                code="rerank_invalid_response",
            )
        return parsed


async def _call_with_retry[T](
    request: Callable[[], T],
    *,
    timeout_seconds: int,
    max_retries: int,
    service_name: str,
) -> T:
    last_message = f"{service_name} 服务调用失败。"
    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(request),
                timeout=timeout_seconds + 5,
            )
        except TimeoutError:
            status_code = 504
            last_message = f"{service_name} 服务调用超时。"
        except Exception as exc:
            status_code = 503
            last_message = f"{service_name} 服务暂时不可用。"
            logger.warning("%s request failed on attempt %s: %s", service_name, attempt, exc)
        else:
            status_code = int(getattr(response, "status_code", 500))
            if status_code == 200:
                request_id = getattr(response, "request_id", None)
                logger.info("%s request completed request_id=%s", service_name, request_id)
                return response
            message = str(getattr(response, "message", "") or "")[:300]
            code = str(getattr(response, "code", "") or "")[:100]
            last_message = f"{service_name} 服务返回错误：{message or code or status_code}"

        if status_code not in _TRANSIENT_STATUS_CODES or attempt == max_retries:
            raise AppError(
                last_message,
                status_code=503 if status_code in _TRANSIENT_STATUS_CODES else 502,
                code=f"{service_name.lower()}_provider_error",
            )
        await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
    raise AppError(last_message, status_code=503, code=f"{service_name.lower()}_provider_error")