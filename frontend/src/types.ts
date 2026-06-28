export interface Paper {
  id: string;
  title: string;
  original_filename: string;
  page_count: number;
  status: "pending" | "processing" | "ready" | "failed";
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
  web_search: string;
}

export interface SseEvent {
  event: string;
  data: unknown;
}

