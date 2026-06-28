import type {
  ChatResponse,
  ConversationDetail,
  ConversationSummary,
  Health,
  Paper,
  SseEvent
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.error?.message ?? `请求失败 (${response.status})`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  papers: () => request<Paper[]>("/papers"),
  conversations: () => request<ConversationSummary[]>("/conversations"),
  conversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deletePaper: (id: string) => request<void>(`/papers/${id}`, { method: "DELETE" }),
  uploadPaper: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ paper: Paper; ingestion_job: { id: string }; duplicated: boolean }>(
      "/papers",
      { method: "POST", body }
    );
  },
  uploadMetrics: async (paperId: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<{ paper_id: string; imported: number }>(
      `/papers/${paperId}/metrics/import`,
      { method: "POST", body }
    );
  }
};

export function parseSseBlock(block: string): SseEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  const raw = dataLines.join("\n");
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: raw };
  }
}

export async function streamChat(
  payload: {
    question: string;
    conversation_id?: string;
    paper_ids: string[];
    enable_web: boolean;
  },
  onEvent: (event: SseEvent) => void
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) {
    throw new Error(`对话请求失败 (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const event = parseSseBlock(block);
      if (event) onEvent(event);
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const event = parseSseBlock(buffer);
    if (event) onEvent(event);
  }
}

export function isChatResponse(value: unknown): value is ChatResponse {
  return Boolean(
    value &&
      typeof value === "object" &&
      "conversation_id" in value &&
      "answer" in value
  );
}

