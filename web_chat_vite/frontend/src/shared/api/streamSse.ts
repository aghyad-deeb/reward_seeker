import { apiUrl } from './client'

export interface SseEventPayload {
  text?: string
  done?: boolean
  error?: string
}

export async function streamJsonSse(
  path: string,
  body: unknown,
  onEvent: (payload: SseEventPayload) => void,
  signal?: AbortSignal,
) {
  const response = await fetch(apiUrl(path), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok || !response.body) {
    let detail = `Request failed: ${response.status}`
    try {
      const text = await response.text()
      const json = JSON.parse(text)
      if (json.detail) detail = `${response.status}: ${json.detail}`
      else if (json.error) detail = `${response.status}: ${json.error}`
      else if (text) detail = `${response.status}: ${text.slice(0, 200)}`
    } catch { /* couldn't read body */ }
    throw new Error(detail)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) {
        continue
      }
      onEvent(JSON.parse(line.slice(6)) as SseEventPayload)
    }
  }
}
