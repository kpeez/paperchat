import { fetchJson } from "./client";

export interface CheckResult {
  ok: boolean;
  message: string | null;
}

export interface BootstrapError {
  code: string;
  message: string;
  action: string;
}

export type BootstrapStatus = "ready" | "starting" | "degraded" | "error";

export interface BootstrapResponse {
  app_version: string;
  status: BootstrapStatus;
  checks: Record<string, CheckResult>;
  errors: BootstrapError[];
}

export interface HealthResponse {
  status: string;
}

export function fetchHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>("/api/health");
}

export function fetchBootstrap(): Promise<BootstrapResponse> {
  return fetchJson<BootstrapResponse>("/api/bootstrap");
}
