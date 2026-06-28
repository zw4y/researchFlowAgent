import math
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)


@dataclass(slots=True)
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(slots=True)
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    name: str

    async def upsert(self, records: list[VectorRecord]) -> None: ...

    async def search(
        self, vector: list[float], paper_ids: list[str], limit: int
    ) -> list[VectorHit]: ...

    async def delete_paper(self, paper_id: str) -> None: ...

    async def close(self) -> None: ...


class InMemoryVectorStore:
    name = "memory"

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, records: list[VectorRecord]) -> None:
        self._records.update({record.id: record for record in records})

    async def search(
        self, vector: list[float], paper_ids: list[str], limit: int
    ) -> list[VectorHit]:
        allowed = set(paper_ids)
        hits: list[VectorHit] = []
        for record in self._records.values():
            if allowed and record.payload.get("paper_id") not in allowed:
                continue
            score = self._cosine(vector, record.vector)
            hits.append(VectorHit(id=record.id, score=score, payload=record.payload))
        return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]

    async def delete_paper(self, paper_id: str) -> None:
        self._records = {
            key: record
            for key, record in self._records.items()
            if record.payload.get("paper_id") != paper_id
        }

    async def close(self) -> None:
        return None

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, url: str, collection: str) -> None:
        self.client = AsyncQdrantClient(url=url)
        self.collection = collection
        self._dimensions: int | None = None

    async def _ensure_collection(self, dimensions: int) -> None:
        if self._dimensions == dimensions:
            return
        exists = await self.client.collection_exists(self.collection)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )
        self._dimensions = dimensions

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        await self._ensure_collection(len(records[0].vector))
        await self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(id=item.id, vector=item.vector, payload=item.payload)
                for item in records
            ],
            wait=True,
        )

    async def search(
        self, vector: list[float], paper_ids: list[str], limit: int
    ) -> list[VectorHit]:
        await self._ensure_collection(len(vector))
        query_filter = None
        if paper_ids:
            query_filter = Filter(
                must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))]
            )
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            VectorHit(id=str(item.id), score=item.score, payload=dict(item.payload or {}))
            for item in response.points
        ]

    async def delete_paper(self, paper_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
            wait=True,
        )

    async def close(self) -> None:
        await self.client.close()
