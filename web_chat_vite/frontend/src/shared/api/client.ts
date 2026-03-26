// In dev, Vite proxies /api requests to the backend — use same-origin (empty).
// In production, set VITE_API_BASE_URL to the backend URL.
const API_BASE_URL = import.meta.env.PROD ? (import.meta.env.VITE_API_BASE_URL || '') : ''

export function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path))
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return (await response.json()) as T
}

export async function postJson<T>(path: string, body: unknown, options?: { signal?: AbortSignal }): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
    signal: options?.signal,
  })
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return (await response.json()) as T
}

export async function putJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return (await response.json()) as T
}

export async function deleteJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: 'DELETE',
  })
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`)
  }
  return (await response.json()) as T
}
