export interface SseEventPayload {
  text?: string
  done?: boolean
  error?: string
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
