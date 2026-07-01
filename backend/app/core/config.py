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

    llm_mode: Literal["openai_compatible"] = "openai_compatible"
    chat_api_key: str | None = None
    chat_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-v4-flash"
    chat_thinking: Literal["default", "enabled", "disabled"] = "disabled"

    embedding_mode: Literal["openai_compatible", "dashscope"] = "dashscope"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = Field(default=1024, ge=64, le=4096)
    embedding_batch_size: int = Field(default=10, ge=1, le=10)
    embedding_timeout_seconds: int = Field(default=30, ge=5, le=180)
    embedding_max_retries: int = Field(default=3, ge=1, le=8)
    embedding_query_instruction: str = (
        "Given a research paper query, retrieve relevant passages that answer the query."
    )
    dashscope_api_key: str | None = None

    rerank_mode: Literal["dashscope"] = "dashscope"
    rerank_model: str = "qwen3-rerank"
    rerank_timeout_seconds: int = Field(default=30, ge=5, le=180)
    rerank_max_retries: int = Field(default=3, ge=1, le=8)
    rerank_instruction: str = (
        "Given a research paper query, retrieve relevant passages that answer the query."
    )
    retrieval_candidates: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=6, ge=1, le=20)

    vector_store_mode: Literal[
        "memory", "qdrant_local", "qdrant_remote", "qdrant"
    ] = "memory"
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "researchflow_chunks"

    tavily_api_key: str | None = None
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    max_pdf_size_mb: int = Field(default=50, ge=1, le=200)
    max_pdf_pages: int = Field(default=500, ge=1, le=2000)
    chunk_size_tokens: int = Field(default=800, ge=100, le=2000)
    chunk_overlap_tokens: int = Field(default=120, ge=0, le=500)
    table_ocr_enabled: bool = True
    table_ocr_dpi: int = Field(default=300, ge=150, le=600)
    table_ocr_min_confidence: float = Field(default=0.5, ge=0, le=1)
    retrieval_score_threshold: float = Field(default=0.0, ge=0, le=1)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def prepare_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite"):
            database_path = self.database_url.rsplit("///", maxsplit=1)[-1]
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()