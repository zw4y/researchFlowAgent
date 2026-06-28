import { describe, expect, it } from "vitest";
import { parseSseBlock } from "./api";

describe("parseSseBlock", () => {
  it("parses named JSON events", () => {
    expect(parseSseBlock('event: token\ndata: {"text":"hello"}')).toEqual({
      event: "token",
      data: { text: "hello" }
    });
  });

  it("returns null for an empty block", () => {
    expect(parseSseBlock("event: ping")).toBeNull();
  });
});

