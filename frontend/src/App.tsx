import {
  BookOpen,
  Check,
  ChevronLeft,
  CircleAlert,
  Database,
  FileSearch,
  Globe2,
  Menu,
  MessageSquare,
  PanelRight,
  Paperclip,
  Plus,
  Send,
  Square,
  Trash2,
  Upload,
  X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, isChatResponse, streamChat } from "./api";
import type {
  ChatResponse,
  Citation,
  ConversationSummary,
  Health,
  Message,
  Paper,
  ToolCall
} from "./types";

type EvidenceTab = "citations" | "tools";

function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationId, setConversationId] = useState<string>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [health, setHealth] = useState<Health>();
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [evidenceTab, setEvidenceTab] = useState<EvidenceTab>("citations");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const uploadInput = useRef<HTMLInputElement>(null);
  const metricsInput = useRef<HTMLInputElement>(null);
  const messagesEnd = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [nextPapers, nextConversations, nextHealth] = await Promise.all([
        api.papers(),
        api.conversations(),
        api.health()
      ]);
      setPapers(nextPapers);
      setConversations(nextConversations);
      setHealth(nextHealth);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "服务连接失败");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (papers.some((paper) => paper.status === "pending" || paper.status === "processing")) {
      const timer = window.setInterval(() => void refresh(), 1500);
      return () => window.clearInterval(timer);
    }
  }, [papers, refresh]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const selectedReadyPapers = useMemo(
    () => papers.filter((paper) => selectedPapers.has(paper.id) && paper.status === "ready"),
    [papers, selectedPapers]
  );

  async function handleUpload(file?: File) {
    if (!file) return;
    setError(undefined);
    try {
      const result = await api.uploadPaper(file);
      setNotice(result.duplicated ? "该论文已在资料库中" : "论文已加入处理队列");
      setSelectedPapers((current) => new Set(current).add(result.paper.id));
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "上传失败");
    } finally {
      if (uploadInput.current) uploadInput.current.value = "";
    }
  }

  async function handleMetricsUpload(file?: File) {
    const paper = selectedReadyPapers[0];
    if (!file || !paper) return;
    try {
      const result = await api.uploadMetrics(paper.id, file);
      setNotice(`已导入 ${result.imported} 条实验指标`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "指标导入失败");
    } finally {
      if (metricsInput.current) metricsInput.current.value = "";
    }
  }

  async function removePaper(paperId: string) {
    try {
      await api.deletePaper(paperId);
      setSelectedPapers((current) => {
        const next = new Set(current);
        next.delete(paperId);
        return next;
      });
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "删除失败");
    }
  }

  function togglePaper(paperId: string) {
    setSelectedPapers((current) => {
      const next = new Set(current);
      if (next.has(paperId)) next.delete(paperId);
      else next.add(paperId);
      return next;
    });
  }

  async function openConversation(id: string) {
    try {
      const detail = await api.conversation(id);
      setConversationId(detail.id);
      setMessages(detail.messages);
      setToolCalls(detail.tool_calls);
      setCitations([]);
      setSidebarOpen(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "加载会话失败");
    }
  }

  function newConversation() {
    setConversationId(undefined);
    setMessages([]);
    setCitations([]);
    setToolCalls([]);
    setQuestion("");
  }

  async function submitQuestion() {
    const prompt = question.trim();
    if (!prompt || streaming) return;
    setQuestion("");
    setError(undefined);
    setNotice(undefined);
    setStreaming(true);
    setCitations([]);
    setToolCalls([]);
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: prompt };
    const assistantId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "" }
    ]);
    try {
      await streamChat(
        {
          question: prompt,
          conversation_id: conversationId,
          paper_ids: [...selectedPapers],
          enable_web: true
        },
        (event) => {
          if (event.event === "token") {
            const token = event.data as { text: string };
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + token.text }
                  : message
              )
            );
          }
          if (event.event === "citation") {
            setCitations((current) => [...current, event.data as Citation]);
            setEvidenceOpen(true);
          }
          if (event.event === "tool.completed") {
            setToolCalls((current) => [...current, event.data as ToolCall]);
          }
          if (event.event === "run.completed" && isChatResponse(event.data)) {
            const result: ChatResponse = event.data;
            setConversationId(result.conversation_id);
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId ? { ...message, content: result.answer } : message
              )
            );
            setCitations(result.citations);
            setToolCalls(result.tool_calls);
          }
          if (event.event === "run.failed") {
            const failure = event.data as { message?: string };
            setError(failure.message ?? "研究工作流执行失败");
          }
        }
      );
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "对话失败");
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(true)} title="论文库">
            <Menu size={19} />
          </button>
          <div className="brand-mark"><FileSearch size={19} /></div>
          <div>
            <strong>ResearchFlow</strong>
            <span>研究工作台</span>
          </div>
        </div>
        <div className="topbar-actions">
          <div className={`service-status ${health?.status ?? "degraded"}`}>
            <span />
            {health?.llm === "fake" ? "本地演示" : health?.status === "ok" ? "服务正常" : "配置不完整"}
          </div>
          <button className="icon-button mobile-only" onClick={() => setEvidenceOpen(true)} title="证据">
            <PanelRight size={19} />
          </button>
          <button className="command-button" onClick={newConversation}>
            <Plus size={16} /> 新研究
          </button>
        </div>
      </header>

      <main className="workspace">
        <aside className={`library-panel ${sidebarOpen ? "open" : ""}`}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">资料库</span>
              <h2>论文</h2>
            </div>
            <button className="icon-button mobile-only" onClick={() => setSidebarOpen(false)} title="关闭">
              <X size={18} />
            </button>
          </div>
          <div className="library-actions">
            <input
              ref={uploadInput}
              hidden
              type="file"
              accept="application/pdf"
              onChange={(event) => void handleUpload(event.target.files?.[0])}
            />
            <button className="primary-action" onClick={() => uploadInput.current?.click()}>
              <Upload size={16} /> 上传论文
            </button>
            <input
              ref={metricsInput}
              hidden
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => void handleMetricsUpload(event.target.files?.[0])}
            />
            <button
              className="icon-button bordered"
              disabled={selectedReadyPapers.length !== 1}
              onClick={() => metricsInput.current?.click()}
              title="导入实验指标 CSV"
            >
              <Database size={17} />
            </button>
          </div>
          <div className="paper-list">
            {papers.length === 0 && <div className="empty-compact">暂无论文</div>}
            {papers.map((paper) => (
              <div className={`paper-row ${selectedPapers.has(paper.id) ? "selected" : ""}`} key={paper.id}>
                <button className="paper-main" onClick={() => togglePaper(paper.id)}>
                  <span className="paper-check">
                    {paper.status === "ready" && selectedPapers.has(paper.id) ? <Check size={13} /> : null}
                    {paper.status === "processing" || paper.status === "pending" ? <span className="spinner" /> : null}
                    {paper.status === "failed" ? <CircleAlert size={13} /> : null}
                  </span>
                  <span className="paper-copy">
                    <strong>{paper.title}</strong>
                    <small>
                      {paper.status === "ready"
                        ? `${paper.page_count} 页`
                        : paper.status === "failed"
                          ? paper.error_message
                          : "正在处理"}
                    </small>
                  </span>
                </button>
                <button className="row-action" onClick={() => void removePaper(paper.id)} title="删除论文">
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
          <div className="conversation-section">
            <div className="section-label">最近研究</div>
            {conversations.map((item) => (
              <button
                className={`conversation-row ${conversationId === item.id ? "active" : ""}`}
                key={item.id}
                onClick={() => void openConversation(item.id)}
              >
                <MessageSquare size={14} />
                <span>{item.title}</span>
              </button>
            ))}
          </div>
        </aside>

        <section className="chat-panel">
          <div className="chat-context">
            <BookOpen size={15} />
            <span>
              {selectedReadyPapers.length
                ? `已选 ${selectedReadyPapers.length} 篇论文`
                : "未选择论文"}
            </span>
            {selectedReadyPapers.slice(0, 2).map((paper) => (
              <button className="context-chip" key={paper.id} onClick={() => togglePaper(paper.id)}>
                {paper.title}<X size={12} />
              </button>
            ))}
          </div>

          <div className="messages" aria-live="polite">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon"><BookOpen size={24} /></div>
                <h1>开始一次研究</h1>
                <p>选择论文，输入你要核对的问题。</p>
              </div>
            )}
            {messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-label">{message.role === "user" ? "你" : "ResearchFlow"}</div>
                <div className="message-content">
                  {message.content || (streaming && message.role === "assistant" ? <span className="typing-indicator"><i /><i /><i /></span> : null)}
                </div>
              </article>
            ))}
            <div ref={messagesEnd} />
          </div>

          {(error || notice) && (
            <div className={`toast ${error ? "error" : "notice"}`}>
              {error ? <CircleAlert size={15} /> : <Check size={15} />}
              <span>{error ?? notice}</span>
              <button onClick={() => { setError(undefined); setNotice(undefined); }} title="关闭"><X size={14} /></button>
            </div>
          )}

          <div className="composer-wrap">
            <div className="composer">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void submitQuestion();
                  }
                }}
                placeholder="输入研究问题"
                rows={2}
              />
              <div className="composer-footer">
                <button className="icon-button" onClick={() => uploadInput.current?.click()} title="上传论文">
                  <Paperclip size={17} />
                </button>
                <div className="composer-modes">
                  <span><Globe2 size={14} /> 联网</span>
                  <span><Database size={14} /> 指标</span>
                </div>
                <button
                  className="send-button"
                  disabled={!question.trim() || streaming}
                  onClick={() => void submitQuestion()}
                  title={streaming ? "正在生成" : "发送"}
                >
                  {streaming ? <Square size={15} fill="currentColor" /> : <Send size={16} />}
                </button>
              </div>
            </div>
          </div>
        </section>

        <aside className={`evidence-panel ${evidenceOpen ? "open" : ""}`}>
          <div className="evidence-header">
            <div className="segmented-control">
              <button className={evidenceTab === "citations" ? "active" : ""} onClick={() => setEvidenceTab("citations")}>
                引用 <span>{citations.length}</span>
              </button>
              <button className={evidenceTab === "tools" ? "active" : ""} onClick={() => setEvidenceTab("tools")}>
                工具 <span>{toolCalls.length}</span>
              </button>
            </div>
            <button className="icon-button mobile-only" onClick={() => setEvidenceOpen(false)} title="关闭">
              <ChevronLeft size={18} />
            </button>
          </div>
          <div className="evidence-content">
            {evidenceTab === "citations" ? (
              citations.length ? citations.map((citation, index) => (
                <article className="citation-item" key={`${citation.chunk_id ?? citation.url}-${index}`}>
                  <div className="citation-meta">
                    <span className={`source-kind ${citation.source_type}`}>
                      {citation.source_type === "paper" ? <BookOpen size={13} /> : <Globe2 size={13} />}
                      {citation.source_type === "paper" ? `第 ${citation.page} 页` : "网页"}
                    </span>
                    {citation.score !== undefined && <span>{Math.round(citation.score * 100)}%</span>}
                  </div>
                  <h3>{citation.paper_title ?? citation.source_title}</h3>
                  <p>{citation.excerpt}</p>
                  {citation.url && <a href={citation.url} target="_blank" rel="noreferrer">打开来源</a>}
                </article>
              )) : <div className="empty-compact">暂无引用</div>
            ) : (
              toolCalls.length ? toolCalls.map((tool) => (
                <article className="tool-item" key={tool.id}>
                  <div className="tool-icon">
                    {tool.status === "completed" ? <Check size={15} /> : <CircleAlert size={15} />}
                  </div>
                  <div>
                    <strong>{tool.name}</strong>
                    <p>{tool.result_summary ?? tool.error_message}</p>
                    <small>{tool.duration_ms} ms</small>
                  </div>
                </article>
              )) : <div className="empty-compact">暂无工具调用</div>
            )}
          </div>
        </aside>
      </main>
      {(sidebarOpen || evidenceOpen) && <button className="mobile-backdrop" onClick={() => { setSidebarOpen(false); setEvidenceOpen(false); }} aria-label="关闭面板" />}
    </div>
  );
}

export default App;

