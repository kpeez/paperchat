import { fetchJson, fetchVoid } from "./client";

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";
export type IngestionJobStatus = "queued" | "running" | "succeeded" | "failed";

export interface IngestionJobResponse {
  id: string;
  document_id: string;
  attempt_number: number;
  status: IngestionJobStatus;
  stage: string | null;
  error_code: string | null;
  error_message: string | null;
}

export interface DocumentResponse {
  id: string;
  content_hash: string;
  original_filename: string;
  display_name: string;
  file_path: string;
  status: DocumentStatus;
  chunk_count: number;
  parser_id: string | null;
  chunker_id: string | null;
  embedding_model_id: string | null;
  error_code: string | null;
  error_message: string | null;
  latest_job: IngestionJobResponse | null;
}

export interface DocumentListResponse {
  documents: DocumentResponse[];
}

export interface DocumentMutationResponse {
  document: DocumentResponse;
  job_enqueued: boolean;
}

export function listDocuments(): Promise<DocumentListResponse> {
  return fetchJson<DocumentListResponse>("/api/documents");
}

export function importDocument(filePath: string): Promise<DocumentMutationResponse> {
  return fetchJson<DocumentMutationResponse>("/api/documents/import", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ file_path: filePath }),
  });
}

export function retryDocument(documentId: string): Promise<DocumentMutationResponse> {
  return fetchJson<DocumentMutationResponse>(`/api/documents/${documentId}/retry`, {
    method: "POST",
  });
}

export function deleteDocument(documentId: string): Promise<void> {
  return fetchVoid(`/api/documents/${documentId}`, {
    method: "DELETE",
  });
}
