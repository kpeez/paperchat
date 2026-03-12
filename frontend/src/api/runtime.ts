import { fetchJson } from "./client";

export interface RuntimeResponse {
  app_version: string;
  data_dir: string;
  database_path: string;
  cache_dir: string;
  embedding_model: string;
}

export function fetchRuntime(): Promise<RuntimeResponse> {
  return fetchJson<RuntimeResponse>("/api/runtime");
}
