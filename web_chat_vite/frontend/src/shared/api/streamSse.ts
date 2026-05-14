import { apiUrl } from './client'

export interface SseEventPayload {
  text?: string
  done?: boolean
  error?: string
  thinking_delta?: string
  text_delta?: string
  tool_calls?: Array<{ type: string; id: string | null; function: { name: string; arguments: string } }>
  content_parts?: Array<{ type: string; text?: string; thinking?: string; summary?: boolean }>
  raw_content?: string  // content with special tokens for rollout_viz compatibility
  openai_response_items?: unknown[]
  structured?: boolean
  generating?: boolean
  turn?: number
  tool_result?: { command: string; output: string }
  max_rounds_reached?: boolean
  parse_error?: boolean
  sampling?: boolean
  attempt?: number
  retry?: boolean
  parse_retry?: number
  max_retries?: number
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
