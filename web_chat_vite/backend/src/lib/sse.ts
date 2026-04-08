export interface SseEventPayload {
  text?: string
  done?: boolean
  error?: string
  // Structured fields (when sidecar is active)
  thinking_delta?: string
  text_delta?: string
  tool_calls?: Array<{ type: string; id: string | null; function: { name: string; arguments: string } }>
  content_parts?: Array<{ type: string; text?: string; thinking?: string }>
  structured?: boolean
}

export function toSseLine(payload: SseEventPayload) {
  return `data: ${JSON.stringify(payload)}\n\n`
}

export function applySseHeaders(res: {
  setHeader(name: string, value: string): void
}) {
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
  res.setHeader('X-Accel-Buffering', 'no')
}
