from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    original_filename: str
    page_count: int
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class IngestionJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    paper_id: str
    status: str
    progress: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class PaperUploadResponse(BaseModel):
    paper: PaperOut
    ingestion_job: IngestionJobOut
    duplicated: bool = False


class CitationOut(BaseModel):
    source_type: Literal["paper", "web"]
    paper_id: str | None = None
    paper_title: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    url: str | None = None
    source_title: str | None = None
    excerpt: str
    score: float | None = None


class ToolCallOut(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["completed", "failed"]
    duration_ms: int
    result_summary: str | None = None
    error_message: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    conversation_id: str | None = None
    paper_ids: list[str] = Field(default_factory=list, max_length=50)
    enable_web: bool = True


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    run_id: str
    answer: str
    citations: list[CitationOut]
    routes: list[str]
    tool_calls: list[ToolCallOut]
    grounding_status: Literal["grounded", "partial", "unsupported"]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    message_metadata: dict[str, Any]
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
    tool_calls: list[ToolCallOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: str
    vector_store: str
    llm: str
    web_search: str


class MetricImportResponse(BaseModel):
    paper_id: str
    imported: int


class StructuredChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    schema_name: Literal["paper_summary", "research_plan"] = "paper_summary"


class StructuredChatResponse(BaseModel):
    data: dict[str, Any]
