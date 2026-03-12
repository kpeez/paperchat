import { fetchJson } from "./client";

export interface PickDocumentsResponse {
  paths: string[];
}

export function pickDocuments(): Promise<PickDocumentsResponse> {
  return fetchJson<PickDocumentsResponse>("/api/local-files/pick-documents", {
    method: "POST",
  });
}
