export interface Paper {
  id: string;
  title: string;
  original_filename: string;
  page_count: number;
  status: "pending" | "processing" | "ready" | "failed";
  index_status: "pending" | "indexing" | "ready" | "stale" | "failed";
  index_profile: string | null;
  indexed_at: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface Citation {
  source_type: "paper" | "web";
  paper_id?: string;
  paper_title?: string;
  page?: number;
  chunk_id?: string;
  url?: string;
  source_title?: string;
  excerpt: string;
  score?: number;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: "completed" | "failed";
  duration_ms: number;
  result_summary?: string;
  error_message?: string;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  run_id: string;
  answer: string;
  citations: Citation[];
  routes: string[];
  tool_calls: ToolCall[];
  grounding_status: "grounded" | "partial" | "unsupported";
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: Message[];
  tool_calls: ToolCall[];
}

export interface Health {
  status: "ok" | "degraded";
  database: string;
  vector_store: string;
  llm: string;
  embedding: string;
  rerank: string;
  index_profile: string;
  web_search: string;
}

export interface IndexStatus {
  provider: string;
  model: string;
  dimensions: number;
  profile_id: string;
  collection: string;
  vector_store_mode: string;
  collection_ready: boolean;
  point_count: number;
  paper_counts: Record<string, number>;
  embedding_configured: boolean;
  rerank_provider: string;
  rerank_model: string;
  rerank_configured: boolean;
}

export interface ReindexResponse {
  paper_id: string;
  job_id: string;
  status: "queued";
}

export interface SseEvent {
  event: string;
  data: unknown;
}
