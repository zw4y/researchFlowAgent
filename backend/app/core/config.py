from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ResearchFlow Agent"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    database_url: str = "sqlite+aiosqlite:///./data/researchflow.db"
    auto_create_tables: bool = True
    upload_dir: Path = Path("./data/uploads")

    llm_mode: Literal["fake", "openai_compatible"] = "fake"
    chat_api_key: str | None = None
    chat_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4.1-mini"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"

    vector_store_mode: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "researchflow_chunks"

    tavily_api_key: str | None = None
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    max_pdf_size_mb: int = Field(default=50, ge=1, le=200)
    max_pdf_pages: int = Field(default=500, ge=1, le=2000)
    chunk_size_tokens: int = Field(default=800, ge=100, le=2000)
    chunk_overlap_tokens: int = Field(default=120, ge=0, le=500)
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    retrieval_score_threshold: float = Field(default=0.25, ge=0, le=1)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def prepare_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite"):
            database_path = self.database_url.rsplit("///", maxsplit=1)[-1]
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
