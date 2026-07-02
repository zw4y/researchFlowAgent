from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RelevantPage(BaseModel):
    paper_title: str
    page: int = Field(ge=1)

    @property
    def key(self) -> str:
        return f"{self.paper_title}:{self.page}"


class EvaluationCase(BaseModel):
    case_id: str
    query: str
    paper_titles: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    relevant_pages: list[RelevantPage] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    answer_type: Literal[
        "factual",
        "numeric_table",
        "architecture",
        "training",
        "ablation",
        "comparison",
        "limitation",
    ] = "factual"
    split: Literal["dev", "test"] = "test"
    label_status: Literal[
        "machine_generated",
        "auto_checked",
        "human_verified",
        "independent_model_verified",
    ] = "machine_generated"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "EvaluationCase":
        if not self.paper_titles and not self.paper_ids:
            raise ValueError("paper_titles or paper_ids must contain at least one paper")
        if not self.relevant_pages and not self.relevant_chunk_ids:
            raise ValueError("at least one relevant page or chunk is required")
        if self.relevant_pages and self.relevant_chunk_ids:
            raise ValueError("use page-level or chunk-level relevance, not both")
        return self


class RankingMetrics(BaseModel):
    precision: float
    recall: float
    reciprocal_rank: float
    hits: int


class CaseResult(BaseModel):
    case_id: str
    query: str
    candidate_k: int
    vector: RankingMetrics
    reranked: RankingMetrics
    candidate_recall: float
    full_context_tokens: int
    vector_context_tokens: int
    reranked_context_tokens: int
    vector_latency_ms: int
    rerank_latency_ms: int
    retrieval_status: Literal["vector", "reranked", "rerank_fallback"]
    vector_keys: list[str] = Field(default_factory=list)
    reranked_keys: list[str] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    case_count: int
    candidate_recall: float
    vector_precision: float
    reranked_precision: float
    precision_delta: float
    vector_recall: float
    reranked_recall: float
    recall_delta: float
    vector_mrr: float
    reranked_mrr: float
    mrr_delta: float
    vector_token_savings: float
    reranked_token_savings: float
    average_vector_latency_ms: float
    average_rerank_latency_ms: float
    rerank_success_rate: float


class EvaluationReport(BaseModel):
    generated_at: datetime
    dataset_path: str
    index_profile: str
    embedding_model: str
    rerank_model: str
    top_k: int
    summary: EvaluationSummary
    cases: list[CaseResult]

