import {
  startTransition,
  useEffect,
  useState,
  type FormEvent,
} from "react";
import { Link, useLocation } from "react-router";
import { ApiError } from "../api/client";
import {
  deleteDocument,
  importDocument,
  listDocuments,
  retryDocument,
  type DocumentResponse,
  type DocumentStatus,
} from "../api/documents";
import { pickDocuments } from "../api/localFiles";
import { fetchRuntime, type RuntimeResponse } from "../api/runtime";
import { ProductNav } from "../components/ProductNav";

type NoticeTone = "success" | "warning" | "error";

interface Notice {
  tone: NoticeTone;
  message: string;
}

const ACTIVE_STATUSES: DocumentStatus[] = ["pending", "processing"];

export function MainPage() {
  const location = useLocation();
  const initialVersion = (location.state as { version?: string })?.version;
  const [runtime, setRuntime] = useState<RuntimeResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [manualPath, setManualPath] = useState("");
  const [notice, setNotice] = useState<Notice | null>(null);
  const [pickerNotice, setPickerNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [addingDocuments, setAddingDocuments] = useState(false);
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);

  async function loadRuntime() {
    try {
      const data = await fetchRuntime();
      startTransition(() => {
        setRuntime(data);
      });
    } catch (error) {
      setNotice({
        tone: "error",
        message: toErrorMessage(error),
      });
    }
  }

  async function loadDocuments(silent = false) {
    if (!silent) {
      setLoading(true);
    }

    try {
      const data = await listDocuments();
      startTransition(() => {
        setDocuments(data.documents);
      });
    } catch (error) {
      if (!silent) {
        setNotice({
          tone: "error",
          message: toErrorMessage(error),
        });
      }
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }

  async function importPaths(paths: string[]) {
    if (paths.length === 0) {
      return;
    }

    const outcome = {
      queued: 0,
      duplicates: 0,
      failedDuplicates: 0,
    };

    for (const path of paths) {
      const result = await importDocument(path);
      if (result.job_enqueued) {
        outcome.queued += 1;
        continue;
      }

      if (result.document.status === "failed") {
        outcome.failedDuplicates += 1;
      } else {
        outcome.duplicates += 1;
      }
    }

    await loadDocuments(true);
    setNotice({
      tone:
        outcome.failedDuplicates > 0 || outcome.duplicates > 0 ? "warning" : "success",
      message: summarizeImportOutcome(outcome),
    });
  }

  useEffect(() => {
    void loadRuntime();
    void loadDocuments();
  }, []);

  const shouldPoll = documents.some((document) =>
    ACTIVE_STATUSES.includes(document.status),
  );

  useEffect(() => {
    if (!shouldPoll) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void loadDocuments(true);
    }, 1500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [shouldPoll]);

  async function handlePickDocuments() {
    setAddingDocuments(true);
    setNotice(null);

    try {
      const result = await pickDocuments();
      if (result.paths.length === 0) {
        return;
      }

      setPickerNotice(null);
      await importPaths(result.paths);
    } catch (error) {
      const message = toErrorMessage(error);
      setPickerNotice(message);
      setNotice({
        tone: "warning",
        message,
      });
    } finally {
      setAddingDocuments(false);
    }
  }

  async function handleManualImport(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedPath = manualPath.trim();
    if (!trimmedPath) {
      return;
    }

    setAddingDocuments(true);
    setNotice(null);
    try {
      await importPaths([trimmedPath]);
      setManualPath("");
    } catch (error) {
      setNotice({
        tone: "error",
        message: toErrorMessage(error),
      });
    } finally {
      setAddingDocuments(false);
    }
  }

  async function handleRetry(document: DocumentResponse) {
    setActiveDocumentId(document.id);
    setNotice(null);
    try {
      const result = await retryDocument(document.id);
      await loadDocuments(true);
      setNotice({
        tone: result.job_enqueued ? "success" : "warning",
        message: result.job_enqueued
          ? `${document.display_name} was queued for another ingestion attempt.`
          : `${document.display_name} is already active or ready.`,
      });
    } catch (error) {
      setNotice({
        tone: "error",
        message: toErrorMessage(error),
      });
    } finally {
      setActiveDocumentId(null);
    }
  }

  async function handleDelete(document: DocumentResponse) {
    const shouldDelete = window.confirm(
      `Delete ${document.display_name} from PaperChat?`,
    );
    if (!shouldDelete) {
      return;
    }

    setActiveDocumentId(document.id);
    setNotice(null);
    try {
      await deleteDocument(document.id);
      await loadDocuments(true);
      setNotice({
        tone: "success",
        message: `${document.display_name} was removed from the library.`,
      });
    } catch (error) {
      setNotice({
        tone: "error",
        message: toErrorMessage(error),
      });
    } finally {
      setActiveDocumentId(null);
    }
  }

  const version = runtime?.app_version ?? initialVersion;
  const readyCount = documents.filter((document) => document.status === "ready").length;
  const failedCount = documents.filter((document) => document.status === "failed").length;
  const activeCount = documents.filter((document) =>
    ACTIVE_STATUSES.includes(document.status),
  ).length;

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(13,148,136,0.16),_transparent_26%),radial-gradient(circle_at_top_right,_rgba(217,119,6,0.14),_transparent_30%),linear-gradient(180deg,_#f8fafc_0%,_#ecfeff_100%)] text-slate-900">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-5 py-6 lg:px-8 lg:py-8">
        <header className="rounded-[2rem] border border-white/70 bg-white/80 px-6 py-6 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.55)] backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-2">
              <div className="space-y-1">
                <p className="text-xs font-semibold uppercase tracking-[0.32em] text-teal-700">
                  Local Research Library
                </p>
                <h1
                  className="text-3xl font-semibold tracking-tight text-slate-950 lg:text-4xl"
                  style={{ fontFamily: '"Iowan Old Style", "Palatino Linotype", Georgia, serif' }}
                >
                  PaperChat
                </h1>
                <p className="max-w-2xl text-sm leading-6 text-slate-600 lg:text-base">
                  Add PDFs in place, monitor ingestion, and keep your local library grounded before
                  chat ever starts guessing.
                </p>
              </div>
              <ProductNav />
            </div>

            <div className="grid grid-cols-3 gap-3 text-sm">
              <SummaryCard label="Ready" value={readyCount} tone="success" />
              <SummaryCard label="Active" value={activeCount} tone="active" />
              <SummaryCard label="Failed" value={failedCount} tone="warning" />
            </div>
          </div>

          {version && (
            <p className="mt-4 text-xs tracking-[0.2em] text-slate-500 uppercase">
              v{version}
            </p>
          )}
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.8fr)_minmax(20rem,1fr)]">
          <section className="rounded-[2rem] border border-white/70 bg-white/80 p-5 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.55)] backdrop-blur lg:p-6">
            <div className="flex flex-col gap-4 border-b border-slate-200/80 pb-5">
              <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
                <div className="space-y-1">
                  <h2 className="text-xl font-semibold text-slate-950">Document library</h2>
                  <p className="text-sm leading-6 text-slate-600">
                    PaperChat keeps references to your existing PDF paths. Re-importing the same
                    bytes reuses the tracked document until you delete it.
                  </p>
                </div>

                <button
                  className="inline-flex items-center justify-center rounded-full bg-slate-950 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                  disabled={addingDocuments}
                  onClick={() => {
                    void handlePickDocuments();
                  }}
                  type="button"
                >
                  {addingDocuments ? "Working..." : "Choose PDFs"}
                </button>
              </div>

              {readyCount > 0 && (
                <Link
                  className="inline-flex items-center justify-center rounded-full border border-slate-200 bg-slate-50 px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-teal-300 hover:text-teal-700 xl:self-start"
                  to="/app/chat"
                >
                  Open grounded chat
                </Link>
              )}

              <form className="grid gap-3 md:grid-cols-[1fr_auto]" onSubmit={handleManualImport}>
                <label className="space-y-2">
                  <span className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
                    Manual path entry
                  </span>
                  <input
                    className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-900 shadow-inner shadow-slate-100 outline-none transition placeholder:text-slate-400 focus:border-teal-500 focus:bg-white"
                    onChange={(event) => {
                      setManualPath(event.target.value);
                    }}
                    placeholder="/absolute/path/to/paper.pdf"
                    value={manualPath}
                  />
                </label>

                <button
                  className="inline-flex items-center justify-center rounded-2xl border border-teal-700/20 bg-teal-600 px-5 py-3 text-sm font-medium text-white transition hover:bg-teal-500 disabled:cursor-not-allowed disabled:bg-slate-400 md:self-end"
                  disabled={addingDocuments || manualPath.trim().length === 0}
                  type="submit"
                >
                  Add by path
                </button>
              </form>

              {notice && <NoticeBanner notice={notice} />}
            </div>

            {loading ? (
              <div className="flex min-h-64 items-center justify-center">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-400">
                  Loading library...
                </p>
              </div>
            ) : documents.length === 0 ? (
              <EmptyState />
            ) : (
              <div className="mt-5 space-y-4">
                {documents.map((document) => (
                  <article
                    className="rounded-[1.5rem] border border-slate-200/80 bg-slate-50/80 p-4 shadow-[0_14px_45px_-38px_rgba(15,23,42,0.7)]"
                    key={document.id}
                  >
                    <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                      <div className="space-y-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-lg font-semibold text-slate-950">
                            {document.display_name}
                          </h3>
                          <StatusBadge
                            stage={document.latest_job?.stage ?? null}
                            status={document.status}
                          />
                        </div>

                        <div className="grid gap-2 text-sm text-slate-600 md:grid-cols-2">
                          <MetaLine label="Path" value={document.file_path} />
                          <MetaLine label="Chunks" value={String(document.chunk_count)} />
                          <MetaLine
                            label="Parser"
                            value={document.parser_id ?? document.latest_job?.stage ?? "queued"}
                          />
                          <MetaLine
                            label="Embedding"
                            value={document.embedding_model_id ?? "pending"}
                          />
                        </div>

                        <p className="font-mono text-xs uppercase tracking-[0.18em] text-slate-400">
                          {shortHash(document.content_hash)}
                        </p>

                        {document.error_message && (
                          <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                            {document.error_message}
                          </div>
                        )}
                      </div>

                      <div className="flex flex-wrap gap-2 xl:justify-end">
                        <button
                          className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-teal-300 hover:text-teal-700 disabled:cursor-not-allowed disabled:opacity-50"
                          disabled={
                            activeDocumentId === document.id || document.status !== "failed"
                          }
                          onClick={() => {
                            void handleRetry(document);
                          }}
                          type="button"
                        >
                          Retry
                        </button>
                        <button
                          className="rounded-full border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-medium text-rose-700 transition hover:border-rose-300 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                          disabled={activeDocumentId === document.id}
                          onClick={() => {
                            void handleDelete(document);
                          }}
                          type="button"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <aside className="space-y-6">
            <section className="rounded-[2rem] border border-white/70 bg-slate-950 px-5 py-6 text-slate-100 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.7)]">
              <div className="space-y-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-teal-300">
                    Runtime
                  </p>
                  <h2 className="mt-2 text-xl font-semibold">Local-first by default</h2>
                </div>

                {runtime ? (
                  <dl className="space-y-3 text-sm leading-6 text-slate-300">
                    <RuntimeLine label="Data directory" value={runtime.data_dir} />
                    <RuntimeLine label="Database" value={runtime.database_path} />
                    <RuntimeLine label="Cache" value={runtime.cache_dir} />
                    <RuntimeLine label="Embedding model" value={runtime.embedding_model} />
                  </dl>
                ) : (
                  <p className="text-sm text-slate-400">Runtime details are loading...</p>
                )}
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/70 bg-white/85 px-5 py-6 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.55)]">
              <div className="space-y-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-amber-700">
                    Add flow
                  </p>
                  <h2 className="mt-2 text-xl font-semibold text-slate-950">
                    Reference PDFs where they already live
                  </h2>
                </div>

                <p className="text-sm leading-6 text-slate-600">
                  Use the native picker when it is available. If your machine cannot provide one,
                  manual path entry stays available so the library never blocks on UI shell quirks.
                </p>

                {pickerNotice && (
                  <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                    {pickerNotice}
                  </div>
                )}
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({
  label,
  tone,
  value,
}: {
  label: string;
  tone: "success" | "active" | "warning";
  value: number;
}) {
  const palette =
    tone === "success"
      ? "border-emerald-200 bg-emerald-50 text-emerald-900"
      : tone === "active"
        ? "border-teal-200 bg-teal-50 text-teal-900"
        : "border-amber-200 bg-amber-50 text-amber-900";

  return (
    <div className={`rounded-2xl border px-4 py-3 ${palette}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.22em]">{label}</p>
      <p className="mt-2 text-2xl font-semibold">{value}</p>
    </div>
  );
}

function StatusBadge({
  stage,
  status,
}: {
  stage: string | null;
  status: DocumentStatus;
}) {
  const palette =
    status === "ready"
      ? "border-emerald-200 bg-emerald-50 text-emerald-800"
      : status === "failed"
        ? "border-rose-200 bg-rose-50 text-rose-700"
        : "border-teal-200 bg-teal-50 text-teal-700";

  return (
    <span className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] ${palette}`}>
      {stage ?? status}
    </span>
  );
}

function MetaLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-1">
      <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">
        {label}
      </p>
      <p className="break-all text-sm text-slate-700">{value}</p>
    </div>
  );
}

function RuntimeLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-1">
      <dt className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
        {label}
      </dt>
      <dd className="break-all text-slate-100">{value}</dd>
    </div>
  );
}

function NoticeBanner({ notice }: { notice: Notice }) {
  const palette =
    notice.tone === "success"
      ? "border-emerald-200 bg-emerald-50 text-emerald-900"
      : notice.tone === "warning"
        ? "border-amber-200 bg-amber-50 text-amber-900"
        : "border-rose-200 bg-rose-50 text-rose-900";

  return (
    <div className={`rounded-2xl border px-4 py-3 text-sm ${palette}`}>
      {notice.message}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex min-h-64 flex-col items-center justify-center gap-3 text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.32em] text-slate-400">
        No documents yet
      </p>
      <h3
        className="text-2xl font-semibold text-slate-950"
        style={{ fontFamily: '"Iowan Old Style", "Palatino Linotype", Georgia, serif' }}
      >
        Start with a PDF that already lives on your machine.
      </h3>
      <p className="max-w-md text-sm leading-6 text-slate-600">
        Choose files through the native picker or paste an absolute path to queue the first local
        ingestion job.
      </p>
    </div>
  );
}

function summarizeImportOutcome(outcome: {
  queued: number;
  duplicates: number;
  failedDuplicates: number;
}) {
  const parts: string[] = [];

  if (outcome.queued > 0) {
    parts.push(
      outcome.queued === 1
        ? "1 document queued for ingestion."
        : `${outcome.queued} documents queued for ingestion.`,
    );
  }

  if (outcome.duplicates > 0) {
    parts.push(
      outcome.duplicates === 1
        ? "1 document was already tracked."
        : `${outcome.duplicates} documents were already tracked.`,
    );
  }

  if (outcome.failedDuplicates > 0) {
    parts.push(
      outcome.failedDuplicates === 1
        ? "1 tracked document is still failed; use Retry to try again."
        : `${outcome.failedDuplicates} tracked documents are still failed; use Retry to try again.`,
    );
  }

  if (parts.length === 0) {
    return "No documents were selected.";
  }

  return parts.join(" ");
}

function shortHash(value: string) {
  return `${value.slice(0, 12)}...${value.slice(-8)}`;
}

function toErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error && error.message.length > 0) {
    return error.message;
  }
  return "Something went wrong while talking to the local PaperChat API.";
}
