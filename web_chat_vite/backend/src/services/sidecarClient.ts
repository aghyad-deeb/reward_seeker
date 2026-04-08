/**
 * HTTP client for the Python renderer sidecar service.
 * All methods return null on connection failure, enabling graceful fallback
 * to regex-based parsing when the sidecar is unavailable.
 */

export interface ContentPart {
  type: 'text' | 'thinking'
  text?: string
  thinking?: string
}

export interface ToolCallPayload {
  type: 'function'
  id: string | null
  function: { name: string; arguments: string }
}

export interface ParsedResponse {
  content_parts: ContentPart[] | null
  content_text: string | null
  tool_calls: ToolCallPayload[]
  unparsed_tool_calls: Array<{ raw_text: string; error: string }>
  parse_success: boolean
  method: 'token_based' | 'text_based' | 'none'
}

interface ToolSpec {
  name: string
  description: string
  parameters: Record<string, unknown>
}

interface SidecarMessage {
  role: string
  content: string | ContentPart[]
  tool_calls?: ToolCallPayload[]
  tool_call_id?: string
  name?: string
}

export class SidecarClient {
  private baseUrl: string
  private available: boolean | null = null
  private lastCheck = 0
  private readonly CHECK_INTERVAL = 30_000 // re-check every 30s

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || `http://localhost:${process.env.SIDECAR_PORT || '8348'}`
  }

  async isAvailable(): Promise<boolean> {
    const now = Date.now()
    if (this.available !== null && now - this.lastCheck < this.CHECK_INTERVAL) {
      return this.available
    }

    try {
      const res = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(2000) })
      this.available = res.ok
    } catch {
      this.available = false
    }
    this.lastCheck = now
    return this.available
  }

  async detectRenderer(modelName: string): Promise<string | null> {
    if (!(await this.isAvailable())) return null

    try {
      const res = await fetch(`${this.baseUrl}/detect-renderer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
        signal: AbortSignal.timeout(5000),
      })
      if (!res.ok) return null
      const data = (await res.json()) as { renderer_name: string | null }
      return data.renderer_name
    } catch {
      return null
    }
  }

  async formatTools(
    rendererName: string,
    modelName: string,
    tools: ToolSpec[],
    systemPrompt: string,
  ): Promise<SidecarMessage[] | null> {
    if (!(await this.isAvailable())) return null

    try {
      const res = await fetch(`${this.baseUrl}/format-tools`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          renderer_name: rendererName,
          model_name: modelName,
          tools,
          system_prompt: systemPrompt,
        }),
        signal: AbortSignal.timeout(10000),
      })
      if (!res.ok) return null
      const data = (await res.json()) as { messages: SidecarMessage[] }
      return data.messages
    } catch {
      return null
    }
  }

  async parseResponse(
    rendererName: string,
    modelName: string,
    responseText: string,
  ): Promise<ParsedResponse | null> {
    if (!(await this.isAvailable())) return null

    try {
      const res = await fetch(`${this.baseUrl}/parse-response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          renderer_name: rendererName,
          model_name: modelName,
          response_text: responseText,
        }),
        signal: AbortSignal.timeout(15000),
      })
      if (!res.ok) return null
      return (await res.json()) as ParsedResponse
    } catch {
      return null
    }
  }

  async *generate(
    rendererName: string,
    modelName: string,
    messages: Array<{ role: string; content: string }>,
    options: {
      maxTokens?: number
      temperature?: number
      seed?: number
      apiKey?: string | null
      baseUrl?: string | null
      toolAddendum?: string | null
    } = {},
  ): AsyncGenerator<string> {
    const body = {
      renderer_name: rendererName,
      model_name: modelName,
      messages,
      max_tokens: options.maxTokens ?? 4096,
      temperature: options.temperature ?? 1,
      seed: options.seed,
      api_key: options.apiKey,
      base_url: options.baseUrl,
      system_prompt_override: options.toolAddendum ? undefined : undefined, // tool addendum handled by renderer
    }

    const res = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(120000),
    })

    if (!res.ok || !res.body) {
      yield `data: ${JSON.stringify({ error: `Sidecar generate failed: ${res.status}` })}\n\n`
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value, { stream: true })
        yield text
      }
    } finally {
      reader.releaseLock()
    }
  }

  async getStopSequences(
    rendererName: string,
    modelName: string,
  ): Promise<{ stop_sequences: string[]; stop_token_ids: number[] } | null> {
    if (!(await this.isAvailable())) return null

    try {
      const res = await fetch(
        `${this.baseUrl}/stop-sequences?renderer_name=${encodeURIComponent(rendererName)}&model_name=${encodeURIComponent(modelName)}`,
        { signal: AbortSignal.timeout(10000) },
      )
      if (!res.ok) return null
      return (await res.json()) as { stop_sequences: string[]; stop_token_ids: number[] }
    } catch {
      return null
    }
  }
}
