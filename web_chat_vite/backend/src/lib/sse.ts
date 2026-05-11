export interface SseEventPayload {
  text?: string
  done?: boolean
  error?: string
  // Status label for non-streaming renderer path (tinker_service /step)
  sampling?: boolean
  // Structured fields emitted on the terminal `done` event for renderer models
  tool_calls?: Array<{ type: string; id: string | null; function: { name: string; arguments: string } }>
  content_parts?: Array<{ type: string; text?: string; thinking?: string }>
  parse_error?: boolean
  // rl_late-only: opaque list of Responses API output items (reasoning,
  // function_call, hosted-tool-call) that the frontend must preserve on
  // the assistant ChatMessage and replay on the next /step so reasoning
  // state + function-call round-trip survive across turns.
  openai_response_items?: unknown[]
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
