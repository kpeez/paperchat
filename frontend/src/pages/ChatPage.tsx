import { startTransition, useEffect, useState, type FormEvent } from "react";
import {
  getConversation,
  listConversations,
  sendMessage,
  type ChatMessageResponse,
  type ConversationResponse,
  type ConversationSummaryResponse,
} from "../api/chat";
import { ApiError } from "../api/client";
import { listDocuments, type DocumentResponse } from "../api/documents";
import { ProductNav } from "../components/ProductNav";

type NoticeTone = "success" | "warning" | "error";

interface Notice {
  tone: NoticeTone;
  message: string;
}

export function ChatPage() {
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [conversations, setConversations] = useState<ConversationSummaryResponse[]>([]);
  const [activeConversation, setActiveConversation] = useState<ConversationResponse | null>(null);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [expandedAssistantMessageId, setExpandedAssistantMessageId] = useState<string | null>(null);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  useEffect(() => {
    void (async () => {
      setLoading(true);
      try {
        const [documentData, conversationData] = await Promise.all([
          listDocuments(),
          listConversations(),
        ]);

        startTransition(() => {
          setDocuments(documentData.documents);
          setConversations(conversationData.conversations);
        });

        const nextConversationId = conversationData.conversations[0]?.id ?? null;
        if (nextConversationId) {
          await loadConversation(nextConversationId);
        } else {
          startTransition(() => {
            setActiveConversation(null);
            setActiveConversationId(null);
            setExpandedAssistantMessageId(null);
          });
        }
      } catch (error) {
        setNotice({
          tone: "error",
          message: toErrorMessage(error),
        });
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    const readyIds = documents
      .filter((document) => document.status === "ready")
      .map((document) => document.id);
    if (readyIds.length === 0) {
      setSelectedDocumentIds([]);
      return;
    }

    setSelectedDocumentIds((currentIds) => {
      const nextIds = currentIds.filter((documentId) => readyIds.includes(documentId));
      if (nextIds.length > 0) {
        return nextIds;
      }
      return readyIds;
    });
  }, [documents]);

  async function loadConversation(conversationId: string) {
    const data = await getConversation(conversationId);
    startTransition(() => {
      setActiveConversation(data);
      setActiveConversationId(data.id);
      setExpandedAssistantMessageId(latestAssistantMessage(data.messages)?.id ?? null);
      setSidebarOpen(false);
    });
  }

  async function refreshConversations() {
    const data = await listConversations();
    startTransition(() => {
      setConversations(data.conversations);
    });
  }

  async function handleSendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      return;
    }

    setSending(true);
    setNotice(null);

    try {
      const result = await sendMessage({
        prompt: trimmedPrompt,
        conversationId: activeConversationId,
        documentIds: selectedDocumentIds,
      });
      await Promise.all([refreshConversations(), loadConversation(result.conversation.id)]);
      setPrompt("");
      setNotice({
        tone: "success",
        message:
          result.source_document_ids.length === readyDocuments.length
            ? "Grounded against all ready documents."
            : `Grounded against ${result.source_document_ids.length} selected document${result.source_document_ids.length === 1 ? "" : "s"}.`,
      });
    } catch (error) {
      setNotice({
        tone: "error",
        message: toErrorMessage(error),
      });
    } finally {
      setSending(false);
    }
  }

  function startNewConversation() {
    setActiveConversation(null);
    setActiveConversationId(null);
    setExpandedAssistantMessageId(null);
    setPrompt("");
    setNotice(null);
    setSidebarOpen(false);
  }

  function toggleDocument(documentId: string) {
    setSelectedDocumentIds((currentIds) => {
      if (currentIds.includes(documentId)) {
        const nextIds = currentIds.filter((id) => id !== documentId);
        return nextIds.length === 0 ? currentIds : nextIds;
      }
      return [...currentIds, documentId];
    });
  }

  const readyDocuments = documents.filter((document) => document.status === "ready");
  const selectedReadyDocuments = readyDocuments.filter((document) =>
    selectedDocumentIds.includes(document.id),
  );

  return (
    <div className="min-h-screen bg-white text-slate-900">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-4 lg:px-6">
        <header className="flex items-center justify-between gap-4 border-b border-slate-200 pb-4">
          <div className="flex min-w-0 items-center gap-4">
            <div>
              <h1
                className="text-2xl font-semibold tracking-tight text-slate-950"
                style={{ fontFamily: '"Iowan Old Style", "Palatino Linotype", Georgia, serif' }}
              >
                PaperChat
              </h1>
              <p className="text-sm text-slate-500">Grounded chat over your local papers.</p>
            </div>
            <div className="hidden md:block">
              <ProductNav />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 lg:hidden"
              onClick={() => {
                setSidebarOpen((current) => !current);
              }}
              type="button"
            >
              {sidebarOpen ? "Close" : "Sidebar"}
            </button>
            <button
              className="rounded-full bg-slate-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800"
              onClick={startNewConversation}
              type="button"
            >
              New chat
            </button>
          </div>
        </header>

        <div className="mt-3 md:hidden">
          <ProductNav />
        </div>

        {notice && <NoticeBanner notice={notice} />}

        <div className="mt-5 grid flex-1 gap-6 lg:grid-cols-[15rem_minmax(0,1fr)]">
          <aside
            className={`${sidebarOpen ? "block" : "hidden"} border-b border-slate-200 pb-4 lg:block lg:border-b-0 lg:border-r lg:pb-0 lg:pr-6`}
          >
            <section>
              <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                Conversations
              </h2>
              <div className="mt-3 space-y-1.5">
                {conversations.length === 0 ? (
                  <p className="text-sm text-slate-500">No conversations yet.</p>
                ) : (
                  conversations.map((conversation) => (
                    <button
                      className={`w-full rounded-xl px-3 py-2 text-left text-sm transition ${
                        activeConversationId === conversation.id
                          ? "bg-slate-100 font-medium text-slate-950"
                          : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                      }`}
                      key={conversation.id}
                      onClick={() => {
                        void loadConversation(conversation.id);
                      }}
                      type="button"
                    >
                      <p className="truncate">{conversation.title ?? "Untitled conversation"}</p>
                      <p className="mt-1 text-xs text-slate-400">
                        {conversation.message_count} messages
                      </p>
                    </button>
                  ))
                )}
              </div>
            </section>

            <section className="mt-6">
              <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                Documents
              </h2>
              <p className="mt-2 text-sm text-slate-500">Choose which ready papers to search.</p>
              <div className="mt-3 space-y-2">
                {readyDocuments.length === 0 ? (
                  <p className="text-sm text-slate-500">
                    Add and finish ingesting a paper in the library first.
                  </p>
                ) : (
                  readyDocuments.map((document) => (
                    <label
                      className="flex items-start gap-3 rounded-xl px-2 py-2 text-sm text-slate-700 hover:bg-slate-50"
                      key={document.id}
                    >
                      <input
                        checked={selectedDocumentIds.includes(document.id)}
                        className="mt-1"
                        onChange={() => {
                          toggleDocument(document.id);
                        }}
                        type="checkbox"
                      />
                      <span className="min-w-0">
                        <span className="block truncate font-medium text-slate-900">
                          {document.display_name}
                        </span>
                        <span className="block text-xs text-slate-400">
                          {document.chunk_count} chunks
                        </span>
                      </span>
                    </label>
                  ))
                )}
              </div>
            </section>
          </aside>

          <main className="flex min-h-[72vh] flex-col">
            <div className="border-b border-slate-200 pb-4">
              <p className="text-base font-medium text-slate-950">
                {activeConversation?.title ?? "New chat"}
              </p>
              <p className="mt-1 text-sm text-slate-500">
                Ask a grounded question about your selected papers.
              </p>
            </div>

            <div className="flex-1 space-y-6 overflow-y-auto py-6">
              {loading ? (
                <div className="flex h-full min-h-64 items-center justify-center">
                  <p className="text-sm text-slate-400">Loading chat...</p>
                </div>
              ) : (activeConversation?.messages ?? []).length === 0 ? (
                <EmptyState />
              ) : (
                (activeConversation?.messages ?? []).map((message) => {
                  const isExpanded = expandedAssistantMessageId === message.id;
                  return (
                    <article className="space-y-3" key={message.id}>
                      <div
                        className={`max-w-3xl ${
                          message.role === "user" ? "ml-auto" : ""
                        }`}
                      >
                        <div className="mb-2 flex items-center gap-3 text-xs text-slate-400">
                          <span className="font-medium uppercase tracking-[0.18em]">
                            {message.role === "user" ? "You" : "Assistant"}
                          </span>
                          <span>{formatDateTime(message.created_at)}</span>
                        </div>

                        {message.role === "assistant" && message.citations ? (
                          <div className="space-y-3">
                            {message.citations.blocks.map((block, index) => (
                              <div
                                className="rounded-2xl bg-slate-50 px-4 py-3 text-sm leading-7 text-slate-700"
                                key={`${message.id}-${index}`}
                              >
                                {block.text}
                              </div>
                            ))}

                            <button
                              className="text-sm font-medium text-slate-500 transition hover:text-slate-900"
                              onClick={() => {
                                setExpandedAssistantMessageId(isExpanded ? null : message.id);
                              }}
                              type="button"
                            >
                              {isExpanded
                                ? "Hide sources"
                                : `Show sources (${message.citations.citations.length})`}
                            </button>

                            {isExpanded && (
                              <div className="space-y-4 border-l border-slate-200 pl-4">
                                {message.citations.citations.map((citation) => (
                                  <article key={citation.citation_id}>
                                    <div className="flex flex-wrap items-center gap-2 text-sm">
                                      <span className="font-medium text-slate-900">
                                        {citation.document_name}
                                      </span>
                                      <span className="text-xs text-slate-400">
                                        {formatPages(citation.page_numbers)}
                                      </span>
                                    </div>
                                    {citation.headings.length > 0 && (
                                      <p className="mt-1 text-xs text-slate-500">
                                        {citation.headings.join(" / ")}
                                      </p>
                                    )}
                                    <p className="mt-2 text-sm leading-6 text-slate-600">
                                      {citation.snippet}
                                    </p>
                                  </article>
                                ))}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="rounded-2xl bg-slate-900 px-4 py-3 text-sm leading-7 text-white">
                            {message.content}
                          </div>
                        )}
                      </div>
                    </article>
                  );
                })
              )}
            </div>

            <form className="border-t border-slate-200 pt-4" onSubmit={handleSendMessage}>
              {selectedReadyDocuments.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">
                  {selectedReadyDocuments.map((document) => (
                    <span
                      className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600"
                      key={document.id}
                    >
                      {document.display_name}
                    </span>
                  ))}
                </div>
              )}

              <label className="block">
                <span className="sr-only">Ask a grounded question</span>
                <textarea
                  className="min-h-28 w-full rounded-2xl border border-slate-200 px-4 py-3 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-slate-400"
                  onChange={(event) => {
                    setPrompt(event.target.value);
                  }}
                  placeholder="Ask a question about your selected papers..."
                  value={prompt}
                />
              </label>

              <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-sm text-slate-500">
                  Searching {selectedDocumentIds.length || readyDocuments.length} ready document
                  {selectedDocumentIds.length === 1 ? "" : "s"}.
                </p>
                <button
                  className="inline-flex items-center justify-center rounded-full bg-slate-950 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                  disabled={sending || readyDocuments.length === 0 || prompt.trim().length === 0}
                  type="submit"
                >
                  {sending ? "Searching..." : "Send"}
                </button>
              </div>
            </form>
          </main>
        </div>
      </div>
    </div>
  );
}

function NoticeBanner({ notice }: { notice: Notice }) {
  const palette =
    notice.tone === "success"
      ? "border-emerald-200 bg-emerald-50 text-emerald-900"
      : notice.tone === "warning"
        ? "border-amber-200 bg-amber-50 text-amber-900"
        : "border-rose-200 bg-rose-50 text-rose-800";

  return (
    <div className={`mt-4 rounded-2xl border px-4 py-3 text-sm ${palette}`}>
      {notice.message}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex min-h-64 items-center justify-center">
      <div className="max-w-md text-center">
        <p className="text-base font-medium text-slate-950">Ready for your first question</p>
        <p className="mt-2 text-sm leading-6 text-slate-500">
          Pick the papers you want to search, then send a prompt. Sources stay tucked away until you
          ask to see them.
        </p>
      </div>
    </div>
  );
}

function latestAssistantMessage(messages: ChatMessageResponse[]): ChatMessageResponse | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") {
      return messages[index] ?? null;
    }
  }
  return null;
}

function formatPages(pageNumbers: number[]) {
  if (pageNumbers.length === 0) {
    return "Pages unknown";
  }
  if (pageNumbers.length === 1) {
    return `Page ${pageNumbers[0]}`;
  }
  return `Pages ${pageNumbers.join(", ")}`;
}

function formatDateTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      });
}

function toErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error && error.message.length > 0) {
    return error.message;
  }
  return "Something went wrong while talking to the chat service.";
}
