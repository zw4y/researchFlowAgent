import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/papers")) return new Response("[]", { status: 200 });
      if (url.endsWith("/conversations")) return new Response("[]", { status: 200 });
      return new Response(
        JSON.stringify({
          status: "ok",
          database: "sqlite",
          vector_store: "memory",
          llm: "fake",
          web_search: "disabled"
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
});

