import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import OpenAI from 'openai'
import type { GenerateRequestBody, OnlineGenerateRequestBody } from '../types/models.js'
import { toSseLine } from '../lib/sse.js'
import {
  BASH_TOOL_SPEC,
  type TinkerServiceClient,
  type TinkerStepMessage,
  type TinkerStepRequest,
} from './tinkerServiceClient.js'

function formatGenerationError(error: unknown): string {
  if (!(error instanceof Error)) return 'Unknown generation error'

  const apiErr = error as unknown as Record<string, unknown>
  const status = apiErr.status

  if (typeof status === 'number') {
    // Try to extract a detailed message from the API response body
    // OpenAI SDK stores parsed body in .error, FastAPI-style servers use .error?.detail or the raw message
    const body = apiErr.error as Record<string, unknown> | undefined
    const detail = body?.detail ?? body?.message ?? (typeof body === 'string' ? body : null)

    if (status === 429) {
      const msg = typeof detail === 'string' ? detail : 'Too many concurrent requests. Please wait a moment and try again.'
      return `Rate limited (429): ${msg}`
    }
    if (status === 401) return 'Authentication failed (401): check your API key.'
    if (status === 403) return 'Access denied (403): your API key may not have access to this model.'
    if (status === 404) return `Model not found (404): ${typeof detail === 'string' ? detail : 'the model ID may be incorrect.'}`
    if (status === 502 || status === 503) return `Server error (${status}): the model server is unavailable. Try again shortly.`
    if (typeof detail === 'string') return `API error (${status}): ${detail}`
    return `API error (${status}): ${error.message}`
  }

  // Connection errors
  if (error.message.includes('ECONNREFUSED')) return 'Connection refused: the model server is not running.'
  if (error.message.includes('ETIMEDOUT') || error.message.includes('timeout')) return 'Request timed out: the model server did not respond in time.'
  if (error.message.includes('Connection error')) return 'Connection error: could not reach the model server.'

  return error.message
}

const DEFAULT_MODEL = 'aptl26/dec22_8b_sdfed'
const DEFAULT_VLLM_URL = 'http://localhost:8901/v1'
const DEFAULT_TINKER_URL = 'https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1'

function projectRoot() {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
}

function vllmEndpointFile() {
  return path.join(projectRoot(), '.vllm_endpoint')
}

export const API_KEY_ENV_VARS: Record<string, string> = {
  openai: 'OPENAI_API_KEY',
  anthropic: 'ANTHROPIC_API_KEY',
  google: 'GOOGLE_API_KEY',
  openrouter: 'OPENROUTER_API_KEY',
  tinker: 'TINKER_API_KEY',
}

type TinkerDispatchProvider = 'rl_late' | 'litellm'

async function readVllmUrlFromFile() {
  try {
    return (await readFile(vllmEndpointFile(), 'utf8')).trim()
  } catch {
    return ''
  }
}

export class GenerationService {
  private cachedVllmUrl: string | null = null
  private cachedClient: OpenAI | null = null
  private tinkerService: TinkerServiceClient | null
  // listTinkerModels() shells out to the `tinker` CLI (~3s: python startup +
  // network auth). The list only changes when a new checkpoint is trained,
  // so cache for a minute to collapse the page-load burst into one call.
  private tinkerModelsCache: { result: { models: string[]; error?: string }; at: number } | null = null
  private static readonly TINKER_MODELS_TTL_MS = 60_000

  constructor(tinkerService?: TinkerServiceClient) {
    this.tinkerService = tinkerService ?? null
  }

  async getVllmBaseUrl() {
    const envUrl = process.env.VLLM_BASE_URL?.trim()
    if (envUrl) {
      return envUrl
    }

    const fileUrl = await readVllmUrlFromFile()
    if (fileUrl) {
      return fileUrl
    }

    return DEFAULT_VLLM_URL
  }

  async setVllmBaseUrl(url: string) {
    const trimmed = url.trim()
    if (!trimmed) {
      throw new Error('URL cannot be empty')
    }
    await writeFile(vllmEndpointFile(), `${trimmed}\n`, 'utf8')
    this.cachedVllmUrl = null
    this.cachedClient = null
    return trimmed
  }

  private async getCachedClient() {
    const url = await this.getVllmBaseUrl()
    if (this.cachedClient && this.cachedVllmUrl === url) {
      return this.cachedClient
    }
    this.cachedVllmUrl = url
    this.cachedClient = new OpenAI({ baseURL: url, apiKey: 'EMPTY' })
    return this.cachedClient
  }

  private async streamOpenAICompatible(request: GenerateRequestBody, baseUrl?: string | null) {
    const messages = request.messages as OpenAI.Chat.ChatCompletionMessageParam[]
    // API key is read from env only. For api.openai.com, use OPENAI_API_KEY.
    // For other OAI-compatible endpoints (vLLM etc.), send 'EMPTY' — those
    // endpoints don't authenticate the key.
    const resolvedKey =
      (baseUrl?.includes('api.openai.com') ? process.env.OPENAI_API_KEY : null) ||
      'EMPTY'
    const client = baseUrl
      ? new OpenAI({ baseURL: baseUrl, apiKey: resolvedKey })
      : await this.getCachedClient()

    const stream = await client.chat.completions.create({
      model: request.model_id ?? DEFAULT_MODEL,
      messages,
      stream: true as const,
      max_tokens: request.max_tokens ?? 4096,
      seed: request.seed ?? 42,
      temperature: request.temperature ?? 1,
    })

    return stream
  }

  async *streamLocal(request: GenerateRequestBody): AsyncGenerator<string> {
    try {
      const modelId = request.model_id ?? DEFAULT_MODEL

      // Explicit provider routing wins: provider presets bypass renderer
      // detection and go straight through tinker_service provider dispatch.
      if (request.provider) {
        if (!this.tinkerService) {
          yield toSseLine({ error: `tinker_service not available (required for ${request.provider} provider)` })
          return
        }
        // Catch the known drift case: frontend persists modelId / baseUrl /
        // provider as separate localStorage keys, and they can fall out of
        // sync (e.g. preset switched but `last-base-url` points to the prior
        // preset's host). rl_late only talks to OpenAI-style /v1/responses;
        // fail fast for the known bad host shapes. litellm intentionally
        // allows custom base URLs, so this guard is rl_late-only.
        if (request.provider === 'rl_late') {
          const badHosts = [
            { host: 'thinkingmachines.dev', name: 'Tinker' },
            { host: 'localhost:8901', name: 'local vLLM' },
          ]
          const bu = request.base_url ?? ''
          const bad = badHosts.find((b) => bu.includes(b.host))
          if (bad) {
            yield toSseLine({
              error:
                `Config mismatch: provider=rl_late requires an OpenAI /v1/responses endpoint `
                + `but base_url="${bu}" points to ${bad.name}. `
                + `Re-select the rl_late model preset to resync the base URL.`,
            })
            return
          }
        }
        yield* this.streamViaTinkerService(request, { provider: request.provider })
        return
      }

      // Implicit routing: if a renderer is available for this model,
      // go through tinker_service's /step — single-turn render→sample→parse
      // matching tinker-cookbook training.
      const rendererName = this.tinkerService ? await this.tinkerService.detectRenderer(modelId) : null
      if (rendererName && this.tinkerService) {
        yield* this.streamViaTinkerService(request, { rendererName })
        return
      }

      // Fallback: direct OAI /chat/completions (vLLM, custom endpoints, no renderer).
      const stream = await this.streamOpenAICompatible(request, request.base_url)
      for await (const chunk of stream) {
        const text = chunk.choices?.[0]?.delta?.content
        if (text) {
          yield toSseLine({ text })
        }
      }
      yield toSseLine({ done: true })
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }

  /**
   * One-shot /step call against tinker_service. Handles both the renderer
   * path (harmony/gpt_oss/qwen/kimi → token-level render→sample→parse) and
   * provider-dispatched streaming paths (rl_late/litellm).
   *
   * Always emits a single terminal `done` event with the decoded message,
   * tool_calls, any `openai_response_items` for rl_late round-trip, and the
   * parse_error flag. That matches what chatCore.ts on the frontend consumes.
   */
  private async *streamViaTinkerService(
    request: GenerateRequestBody,
    opts: { rendererName: string } | { provider: TinkerDispatchProvider },
  ): AsyncGenerator<string> {
    if (!this.tinkerService) {
      yield toSseLine({ error: 'tinker_service not available' })
      return
    }

    // Surface a progress label so the UI can show "⏳ Generating..." while we
    // wait for the (non-streaming) /step call.
    yield toSseLine({ sampling: true })

    const isDispatchedProvider = 'provider' in opts
    const stepReq: TinkerStepRequest = isDispatchedProvider
      ? {
          provider: opts.provider,
          // renderer_name is unused for provider-dispatched paths.
          renderer_name: '',
          model_name: request.model_id ?? DEFAULT_MODEL,
          base_url: request.base_url ?? undefined,
          // api_key omitted — the service reads provider keys from its env.
          messages: request.messages as TinkerStepMessage[],
          // Provider-dispatched paths use native tool specs. Pass the tinker
          // enum value for schema compatibility.
          target_tool_format: 'tinker' as const,
          // Send the bash function spec so the model emits a structured
          // `function_call` item. Without this, the model has no declared
          // bash capability and produces plain prose that nothing executes.
          tools: [BASH_TOOL_SPEC],
          sampling: {
            max_tokens: request.max_tokens ?? 4096,
            ...(opts.provider === 'litellm'
              ? {
                  temperature: request.temperature ?? 1,
                  seed: request.seed ?? undefined,
                  reasoning_effort: request.reasoning_effort ?? undefined,
                }
              : {
                  // rl_late drops temperature/seed/stop silently upstream.
                  // `reasoning_effort` defaults to 'low' (cheap + bash-task
                  // optimal per docs/o3-step41-redwood-visible-cot.md §10) but
                  // is user-overridable via the Reasoning header dropdown.
                  reasoning_effort: request.reasoning_effort ?? 'low',
                }),
          },
        }
      : {
          model_name: request.model_id ?? DEFAULT_MODEL,
          renderer_name: opts.rendererName,
          base_url: request.base_url ?? undefined,
          messages: request.messages as TinkerStepMessage[],
          target_tool_format: 'tinker' as const,
          tools: [BASH_TOOL_SPEC],
          sampling: {
            max_tokens: request.max_tokens ?? 4096,
            temperature: request.temperature ?? 1,
            seed: request.seed ?? undefined,
          },
        }

    try {
      // Provider-dispatched paths stream token-by-token. Tinker (renderer) is
      // non-streaming — it lumps render→sample→parse into one blocking
      // `/step` that returns the whole turn at once.
      if (isDispatchedProvider) {
        yield* this.#streamTinkerServiceProvider(stepReq)
      } else {
        const result = await this.tinkerService.step(stepReq)
        yield toSseLine({
          done: true,
          text: result.decoded_message.content,
          content_parts: result.decoded_message.content_parts ?? undefined,
          tool_calls: result.decoded_message.tool_calls?.length
            ? result.decoded_message.tool_calls
            : undefined,
          parse_error: !result.parse_success,
        })
      }
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }

  /**
   * Consume tinker_service provider SSE streams (rl_late/litellm) and
   * translate each upstream event into the frontend's SSE event shape.
   */
  async *#streamTinkerServiceProvider(stepReq: TinkerStepRequest): AsyncGenerator<string> {
    if (!this.tinkerService) {
      yield toSseLine({ error: 'tinker_service not available' })
      return
    }
    for await (const evt of this.tinkerService.stepStream(stepReq)) {
      const data = (evt.data ?? {}) as Record<string, unknown>
      switch (evt.type) {
        case 'response.output_text.delta': {
          // Visible-answer tokens — forward verbatim; frontend accumulates
          // into `streamed` and fires onStreamingText on each delta.
          const text = typeof data.text === 'string' ? data.text : ''
          if (text) yield toSseLine({ text })
          break
        }
        case 'response.reasoning.delta': {
          // Reasoning arrives as whole chunks (not token-level). Wrap each
          // as its own `<think>…</think>` so the frontend's existing
          // thinking-block renderer picks them up and the user never sees
          // a partial tag. Multiple chunks per turn = multiple tags.
          const text = typeof data.text === 'string' ? data.text : ''
          if (text) yield toSseLine({ text: `<think>${text}</think>` })
          break
        }
        case 'response.hosted_tool.delta': {
          // function_call / web_search_call / code_interpreter_call items
          // arrive whole from upstream. No frontend delta needed — the
          // terminal `done` event carries them via tool_calls +
          // openai_response_items. Buffered implicitly by tinker_service
          // and emitted in its `response.done` payload; we just ignore
          // these on the forwarding side.
          break
        }
        case 'response.done': {
          const decoded = (data.decoded_message ?? {}) as {
            content?: string
            content_parts?: unknown
            tool_calls?: unknown[]
            openai_response_items?: unknown[] | null
          }
          yield toSseLine({
            done: true,
            text: typeof decoded.content === 'string' ? decoded.content : '',
            content_parts: decoded.content_parts as never,
            tool_calls: Array.isArray(decoded.tool_calls) && decoded.tool_calls.length > 0
              ? (decoded.tool_calls as never)
              : undefined,
            openai_response_items: decoded.openai_response_items ?? undefined,
            parse_error: data.parse_success === false,
          })
          return
        }
        case 'response.error': {
          const message = typeof data.message === 'string'
            ? data.message
            : `tinker_service stream error: ${JSON.stringify(data).slice(0, 200)}`
          yield toSseLine({ error: message })
          return
        }
        default:
          // Unknown event — ignore. Future tinker_service versions may add
          // more event types; we should fall forward cleanly.
          break
      }
    }
    // If we get here the stream ended without a `response.done` — treat
    // as error so the retry/display logic sees a terminal state.
    yield toSseLine({ error: 'tinker_service stream ended without response.done' })
  }

  async listModels() {
    try {
      const client = await this.getCachedClient()
      const models = await client.models.list()
      return { models: models.data.map((item) => item.id) }
    } catch (error) {
      return {
        models: [DEFAULT_MODEL],
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  async listEndpointModels(baseUrl: string, apiKey = '') {
    try {
      const client = new OpenAI({ baseURL: baseUrl, apiKey: apiKey || 'EMPTY' })
      // Cap at 10s so a bad user-entered baseUrl doesn't hang the UI for
      // the OpenAI SDK's default ~60s connect timeout.
      const models = await client.models.list({ signal: AbortSignal.timeout(10_000) })
      return { models: models.data.map((item) => item.id) }
    } catch (error) {
      return {
        models: [],
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  getPresets(vllmUrl: string) {
    // API keys are deliberately NOT included in this response — the backend
    // holds them in its own env and never exposes them to the browser.
    const presets: Array<{ id: string; label: string; baseUrl: string }> = [
      { id: 'vllm', label: 'vLLM', baseUrl: vllmUrl },
    ]
    if (process.env.TINKER_API_KEY) {
      presets.push({
        id: 'tinker',
        label: 'Tinker',
        baseUrl: process.env.TINKER_BASE_URL || DEFAULT_TINKER_URL,
      })
    }
    presets.push({ id: 'custom', label: 'Custom', baseUrl: '' })
    return { presets }
  }

  async health() {
    const vllmUrl = await this.getVllmBaseUrl()

    // Probe vLLM with a short timeout so `/api/health` stays fast when the
    // server isn't running (the frontend polls this every 15s for the
    // "Connected" indicator, so a 1.2s default connect-timeout from the
    // OpenAI SDK was dominating page-load wall time).
    const vllmConnected = await Promise.race([
      (async () => {
        try {
          const client = await this.getCachedClient()
          await client.models.list()
          return true
        } catch {
          return false
        }
      })(),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 400)),
    ])

    return {
      status: 'ok',
      vllm_connected: vllmConnected,
      vllm_url: vllmUrl,
      sandbox_endpoint: process.env.SANDBOX_FUSION_ENDPOINT ?? 'http://localhost:60808',
      tinker_service_available: this.tinkerService ? await this.tinkerService.isAvailable() : false,
    }
  }

  async getToolAddendum(
    modelId: string,
    systemPrompt: string,
    rendererOverride?: string,
  ): Promise<{ renderer_name: string | null; addendum: string | null }> {
    if (!this.tinkerService) return { renderer_name: null, addendum: null }

    const rendererName = rendererOverride ?? (await this.tinkerService.detectRenderer(modelId))
    if (!rendererName) return { renderer_name: null, addendum: null }

    const formatted = await this.tinkerService.formatTools({
      model_name: modelId,
      renderer_name: rendererName,
      tools: [BASH_TOOL_SPEC],
      system_prompt: systemPrompt,
    })
    if (!formatted || !formatted.supported) {
      return { renderer_name: rendererName, addendum: null }
    }

    const addendum = formatted.addendum?.trim() || null
    if (!addendum) return { renderer_name: rendererName, addendum: null }

    // Strip the original prompt text if the renderer concatenated it (Qwen3-style);
    // renderers that restructure (GPT-OSS) will still show the wrapping that was added.
    if (systemPrompt && addendum.startsWith(systemPrompt)) {
      const trimmed = addendum.slice(systemPrompt.length).trim()
      return { renderer_name: rendererName, addendum: trimmed || null }
    }
    if (systemPrompt && addendum.includes(systemPrompt)) {
      const trimmed = addendum.replace(systemPrompt, '').trim()
      return { renderer_name: rendererName, addendum: trimmed || null }
    }
    return { renderer_name: rendererName, addendum }
  }

  async detectRenderer(modelId: string): Promise<string | null> {
    if (!this.tinkerService) return null
    return await this.tinkerService.detectRenderer(modelId)
  }

  async checkApiKey(provider: string) {
    if (provider === 'litellm') {
      return { available: true }
    }
    const envVar = API_KEY_ENV_VARS[provider]
    if (!envVar) {
      return { available: false, error: 'Unknown provider' }
    }

    return { available: Boolean(process.env[envVar]) }
  }

  async listProviderModels(provider: string): Promise<{ models: string[]; error?: string }> {
    switch (provider) {
      case 'openai':
        return {
          models: [
            'gpt-4.5-preview',
            'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
            'gpt-4o', 'gpt-4o-mini',
            'o3', 'o3-mini', 'o4-mini',
            'codex-mini-latest',
          ],
        }
      case 'anthropic':
        return {
          models: [
            'claude-opus-4-20250514', 'claude-opus-4-6',
            'claude-sonnet-4-20250514', 'claude-sonnet-4-6',
            'claude-haiku-4-5-20251001',
            'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022',
          ],
        }
      case 'google':
        return {
          models: [
            'gemini-2.5-pro', 'gemini-2.5-flash',
            'gemini-2.0-flash', 'gemini-2.0-flash-lite',
            'gemini-1.5-pro', 'gemini-1.5-flash',
          ],
        }
      case 'openrouter': {
        const apiKey = process.env[API_KEY_ENV_VARS['openrouter'] ?? '']
        if (!apiKey) return { models: [], error: 'OPENROUTER_API_KEY not configured' }
        try {
          // 10s cap — without this a hung OpenRouter endpoint would
          // block the caller for the fetch-default ~30–120s.
          const response = await fetch('https://openrouter.ai/api/v1/models', {
            headers: { Authorization: `Bearer ${apiKey}` },
            signal: AbortSignal.timeout(10_000),
          })
          if (!response.ok) return { models: [], error: `OpenRouter API error: ${response.status}` }
          const data = (await response.json()) as { data: Array<{ id: string }> }
          return { models: data.data.map((m) => m.id) }
        } catch (error) {
          return { models: [], error: error instanceof Error ? error.message : 'Failed to fetch models' }
        }
      }
      case 'tinker':
        return await this.listTinkerModels()
      case 'litellm':
        return {
          models: [
            'openai/gpt-4o',
            'openai/gpt-4.1',
            'anthropic/claude-sonnet-4-6',
            'gemini/gemini-2.5-flash',
            'openrouter/openai/gpt-4o',
          ],
        }
      default:
        return { models: [], error: `Unknown provider: ${provider}` }
    }
  }

  async listTinkerModels() {
    // Serve from cache if fresh — the CLI/API call is ~3s and the list only
    // changes when a new checkpoint is trained, so a 60s TTL turns the
    // page-load burst (model-presets + tinker-models + AddModel form) into
    // one real call followed by cache hits.
    if (
      this.tinkerModelsCache
      && Date.now() - this.tinkerModelsCache.at < GenerationService.TINKER_MODELS_TTL_MS
    ) {
      return this.tinkerModelsCache.result
    }

    const result = await this.#fetchTinkerModels()
    this.tinkerModelsCache = { result, at: Date.now() }
    return result
  }

  async #fetchTinkerModels(): Promise<{ models: string[]; error?: string }> {
    // Try the tinker CLI first — it returns ALL checkpoints with pagination
    try {
      const { execSync } = await import('node:child_process')
      const raw = execSync('tinker --format json checkpoint list --limit=0', {
        timeout: 15000,
        encoding: 'utf-8',
        env: { ...process.env },
      })
      const jsonStart = raw.indexOf('{')
      if (jsonStart >= 0) {
        const data = JSON.parse(raw.slice(jsonStart)) as { checkpoints: Array<{ tinker_path: string; checkpoint_type: string; time: string }> }
        const sampler = data.checkpoints
          .filter((c) => c.checkpoint_type === 'sampler')
          .sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime())
          .map((c) => c.tinker_path)
        if (sampler.length > 0) return { models: sampler }
      }
    } catch {
      // CLI not available or failed — fall back to OpenAI API
    }

    const apiKey = process.env.TINKER_API_KEY
    if (!apiKey) {
      return { models: [], error: 'TINKER_API_KEY not configured' }
    }

    try {
      const client = new OpenAI({
        baseURL: process.env.TINKER_BASE_URL || DEFAULT_TINKER_URL,
        apiKey,
      })
      const models = await client.models.list()
      const sorted = [...models.data].sort((a, b) => (b.created ?? 0) - (a.created ?? 0))
      return { models: sorted.map((item) => item.id) }
    } catch (error) {
      return {
        models: [],
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  private static shouldUseOpenAIResponses(model: string): boolean {
    const normalized = model.toLowerCase()
    return normalized.includes('codex')
      || /^o\d/.test(normalized)
      || normalized.startsWith('gpt-5')
  }

  private static litellmModelName(provider: string, model: string): string {
    if (provider === 'litellm') return model
    if (provider === 'openrouter') {
      return model.startsWith('openrouter/') ? model : `openrouter/${model}`
    }
    if (provider === 'google') {
      return model.startsWith('gemini/') || model.startsWith('vertex_ai/')
        ? model
        : `gemini/${model}`
    }
    if (provider === 'anthropic') {
      return model.startsWith('anthropic/') ? model : `anthropic/${model}`
    }
    if (provider === 'openai') {
      return model.startsWith('openai/') ? model : `openai/${model}`
    }
    return model
  }

  private async buildOnlineTinkerStepRequest(
    request: OnlineGenerateRequestBody,
    apiKey?: string,
  ): Promise<TinkerStepRequest> {
    const provider = request.provider

    if (provider === 'tinker') {
      if (!this.tinkerService) throw new Error('tinker_service not available')
      const rendererName = await this.tinkerService.detectRenderer(request.model)
      if (!rendererName) {
        throw new Error(`No tinker renderer detected for online model "${request.model}"`)
      }
      return {
        provider: 'tinker',
        model_name: request.model,
        renderer_name: rendererName,
        base_url: process.env.TINKER_BASE_URL || DEFAULT_TINKER_URL,
        api_key: apiKey,
        messages: request.messages as TinkerStepMessage[],
        target_tool_format: 'tinker',
        tools: [BASH_TOOL_SPEC],
        sampling: {
          max_tokens: request.max_tokens ?? 4096,
          temperature: request.temperature ?? 1,
        },
      }
    }

    const dispatchProvider: TinkerDispatchProvider =
      provider === 'openai' && GenerationService.shouldUseOpenAIResponses(request.model)
        ? 'rl_late'
        : 'litellm'

    return {
      provider: dispatchProvider,
      model_name: dispatchProvider === 'rl_late'
        ? request.model
        : GenerationService.litellmModelName(provider, request.model),
      renderer_name: '',
      base_url: dispatchProvider === 'rl_late' ? process.env.OPENAI_BASE_URL : undefined,
      api_key: apiKey,
      messages: request.messages as TinkerStepMessage[],
      target_tool_format: 'tinker',
      tools: [BASH_TOOL_SPEC],
      sampling: {
        max_tokens: request.max_tokens ?? 4096,
        ...(dispatchProvider === 'litellm'
          ? { temperature: request.temperature ?? 1 }
          : { reasoning_effort: 'low' as const }),
      },
    }
  }

  private async *streamOnlineViaTinkerService(
    request: OnlineGenerateRequestBody,
    apiKey?: string,
  ): AsyncGenerator<string> {
    if (!this.tinkerService) {
      yield toSseLine({ error: 'tinker_service not available' })
      return
    }

    yield toSseLine({ sampling: true })

    try {
      const stepReq = await this.buildOnlineTinkerStepRequest(request, apiKey)
      if (stepReq.provider === 'tinker') {
        const result = await this.tinkerService.step(stepReq)
        yield toSseLine({
          done: true,
          text: result.decoded_message.content,
          content_parts: result.decoded_message.content_parts ?? undefined,
          tool_calls: result.decoded_message.tool_calls?.length
            ? result.decoded_message.tool_calls
            : undefined,
          parse_error: !result.parse_success,
        })
        return
      }
      yield* this.#streamTinkerServiceProvider(stepReq)
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }

  async *streamOnline(request: OnlineGenerateRequestBody): AsyncGenerator<string> {
    const knownProviders = new Set(['openai', 'anthropic', 'google', 'openrouter', 'tinker', 'litellm'])
    if (!knownProviders.has(request.provider)) {
      yield toSseLine({ error: 'Unknown provider' })
      return
    }

    const envVar = API_KEY_ENV_VARS[request.provider]
    const apiKey = envVar ? process.env[envVar] : undefined
    if (envVar && !apiKey) {
      yield toSseLine({ error: `API key not found for ${request.provider}` })
      return
    }

    if (!this.tinkerService) {
      yield toSseLine({ error: 'tinker_service not available' })
      return
    }

    try {
      for await (const event of this.streamOnlineViaTinkerService(request, apiKey)) {
        yield event
      }
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }
}
