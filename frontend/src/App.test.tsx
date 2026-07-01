import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

let webSearchStatus: "enabled" | "disabled" = "disabled";

afterEach(cleanup);

beforeEach(() => {
  webSearchStatus = "disabled";
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/papers")) return new Response("[]", { status: 200 });
      if (url.endsWith("/conversations")) return new Response("[]", { status: 200 });
      if (url.endsWith("/index/status")) {
        return new Response(
          JSON.stringify({
            provider: "dashscope",
            model: "text-embedding-v4",
            dimensions: 64,
            profile_id: "test-profile",
            collection: "test-collection",
            vector_store_mode: "memory",
            collection_ready: false,
            point_count: 0,
            paper_counts: {},
            embedding_configured: true,
            rerank_provider: "dashscope",
            rerank_model: "qwen3-rerank",
            rerank_configured: true
          }),
          { status: 200 }
        );
      }
      return new Response(
        JSON.stringify({
          status: "ok",
          database: "sqlite",
          vector_store: "memory",
          llm: "openai_compatible",
          embedding: "dashscope",
          rerank: "dashscope",
          index_profile: "test-profile",
          web_search: webSearchStatus
        }),
        { status: 200 }
      );
    })
  );
});

describe("App", () => {
  it("renders the research workspace", async () => {
    render(<App />);
    expect(screen.getByText("ResearchFlow")).toBeInTheDocument();
    expect(await screen.findByText("开始一次研究")).toBeInTheDocument();
  });

  it("disables web search when Tavily is not configured", async () => {
    render(<App />);
    const toggle = await screen.findByRole("button", { name: "联网搜索" });
    expect(toggle).toBeDisabled();
    expect(toggle).toHaveAttribute("aria-pressed", "false");
  });

  it("allows web search to be toggled when Tavily is configured", async () => {
    webSearchStatus = "enabled";
    render(<App />);
    const toggle = await screen.findByRole("button", { name: "联网搜索" });
    await waitFor(() => {
      expect(toggle).toBeEnabled();
      expect(toggle).toHaveAttribute("aria-pressed", "true");
    });
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "false");
  });
});

