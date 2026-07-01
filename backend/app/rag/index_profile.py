import hashlib
import json
from dataclasses import dataclass

from app.core.config import Settings
from app.providers.base import EmbeddingProvider

SPLITTER_VERSION = "llamaindex-sentence-v1"
TABLE_OCR_VERSION = "rapidocr-table-v2"


@dataclass(frozen=True, slots=True)
class IndexProfile:
    provider: str
    model: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int
    splitter_version: str
    table_ocr_enabled: bool
    table_ocr_dpi: int
    table_ocr_version: str
    profile_id: str
    collection_name: str

    @classmethod
    def build(cls, settings: Settings, provider: EmbeddingProvider) -> "IndexProfile":
        values = {
            "provider": provider.name,
            "model": provider.model_name,
            "dimensions": provider.dimensions,
            "chunk_size": settings.chunk_size_tokens,
            "chunk_overlap": settings.chunk_overlap_tokens,
            "splitter_version": SPLITTER_VERSION,
            "table_ocr_enabled": settings.table_ocr_enabled,
            "table_ocr_dpi": settings.table_ocr_dpi,
            "table_ocr_version": TABLE_OCR_VERSION,
        }
        serialized = json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        profile_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        collection_name = f"{settings.qdrant_collection}_{profile_id}"
        return cls(
            provider=provider.name,
            model=provider.model_name,
            dimensions=provider.dimensions,
            chunk_size=settings.chunk_size_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
            splitter_version=SPLITTER_VERSION,
            table_ocr_enabled=settings.table_ocr_enabled,
            table_ocr_dpi=settings.table_ocr_dpi,
            table_ocr_version=TABLE_OCR_VERSION,
            profile_id=profile_id,
            collection_name=collection_name,
        )