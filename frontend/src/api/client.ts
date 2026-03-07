const BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:9712";

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}
