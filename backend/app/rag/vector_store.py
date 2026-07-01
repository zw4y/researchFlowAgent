from collections.abc import Sequence
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.core.config import Settings
from app.providers.base import EmbeddingProvider
from app.rag.index_profile import IndexProfile


class ProviderEmbeddingAdapter(BaseEmbedding):
    _provider: EmbeddingProvider = PrivateAttr()

    def __init__(self, provider: EmbeddingProvider, batch_size: int) -> None:
        super().__init__(model_name=provider.model_name, embed_batch_size=batch_size)
        self._provider = provider

    def _get_query_embedding(self, query: str) -> list[float]:
        raise RuntimeError("ResearchFlow 仅使用异步 LlamaIndex Embedding 接口。")

    def _get_text_embedding(self, text: str) -> list[float]:
        raise RuntimeError("ResearchFlow 仅使用异步 LlamaIndex Embedding 接口。")

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._provider.embed_query(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._provider.embed_documents([text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed_documents(texts)


class LlamaIndexVectorStore:
    def __init__(
        self,
        settings: Settings,
        profile: IndexProfile,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.settings = settings
        self.profile = profile
        self.name = settings.vector_store_mode
        self.client = self._build_client(settings)
        self.embedding_adapter = ProviderEmbeddingAdapter(
            embedding_provider,
            settings.embedding_batch_size,
        )
        self.store = QdrantVectorStore(
            collection_name=profile.collection_name,
            aclient=self.client,
            dense_config=VectorParams(size=profile.dimensions, distance=Distance.COSINE),
            batch_size=64,
            index_doc_id=True,
        )
        self.index = VectorStoreIndex.from_vector_store(
            self.store,
            embed_model=self.embedding_adapter,
        )

    @staticmethod
    def _build_client(settings: Settings) -> AsyncQdrantClient:
        if settings.vector_store_mode == "memory":
            return AsyncQdrantClient(location=":memory:")
        if settings.vector_store_mode == "qdrant_local":
            return AsyncQdrantClient(path=str(settings.qdrant_path))
        return AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )

    async def add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        if nodes:
            await self.index.ainsert_nodes(nodes)

    async def retrieve(
        self,
        query: str,
        paper_ids: list[str],
        limit: int,
    ) -> list[NodeWithScore]:
        filters: list[MetadataFilter] = [
            MetadataFilter(key="index_profile", value=self.profile.profile_id)
        ]
        if paper_ids:
            filters.append(
                MetadataFilter(
                    key="paper_id",
                    value=paper_ids,
                    operator=FilterOperator.IN,
                )
            )
        retriever = self.index.as_retriever(
            similarity_top_k=limit,
            filters=MetadataFilters(filters=filters, condition=FilterCondition.AND),
        )
        return await retriever.aretrieve(query)

    async def delete_paper(self, paper_id: str) -> None:
        response = await self.client.get_collections()
        prefix = f"{self.settings.qdrant_collection}_"
        for collection in response.collections:
            if not collection.name.startswith(prefix):
                continue
            await self.client.delete(
                collection_name=collection.name,
                points_selector=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
                wait=True,
            )

    async def current_collection_exists(self) -> bool:
        return await self.client.collection_exists(self.profile.collection_name)

    async def current_point_count(self) -> int:
        if not await self.current_collection_exists():
            return 0
        info = await self.client.get_collection(self.profile.collection_name)
        return int(info.points_count or 0)

    async def close(self) -> None:
        await self.client.close()

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.name,
            "collection": self.profile.collection_name,
            "profile_id": self.profile.profile_id,
            "dimensions": self.profile.dimensions,
        }