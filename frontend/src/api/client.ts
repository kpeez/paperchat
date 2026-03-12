const BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:9712";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function fetchJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init);
  if (!response.ok) {
    throw await toApiError(response);
  }
  return response.json() as Promise<T>;
}

export async function fetchVoid(path: string, init?: RequestInit): Promise<void> {
  const response = await fetch(`${BASE_URL}${path}`, init);
  if (!response.ok) {
    throw await toApiError(response);
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  let message = `API error: ${response.status}`;

  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.length > 0) {
      message = body.detail;
    }
  } catch {
    // Keep the fallback status message when the response body is not JSON.
  }

  return new ApiError(response.status, message);
}
