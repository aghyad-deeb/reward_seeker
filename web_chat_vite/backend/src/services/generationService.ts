import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import Anthropic from '@anthropic-ai/sdk'
import { GoogleGenAI } from '@google/genai'
import OpenAI from 'openai'
import type { GenerateRequestBody, OnlineGenerateRequestBody } from '../types/models.js'
import { toSseLine, type SseEventPayload } from '../lib/sse.js'

function formatGenerationError(error: unknown): string {
  if (!(error instanceof Error)) return 'Unknown generation error'

  const apiErr = error as Record<string, unknown>
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

  private async streamOpenAICompatible(request: GenerateRequestBody, baseUrl?: string | null, apiKey?: string | null) {
    const messages = request.messages as OpenAI.Chat.ChatCompletionMessageParam[]
    const client = baseUrl
      ? new OpenAI({ baseURL: baseUrl, apiKey: apiKey || 'EMPTY' })
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
      const stream = await this.streamOpenAICompatible(request, request.base_url, request.api_key)
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
      const models = await client.models.list()
      return { models: models.data.map((item) => item.id) }
    } catch (error) {
      return {
        models: [],
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  getPresets(vllmUrl: string) {
    const presets = [
      { id: 'vllm', label: 'vLLM', baseUrl: vllmUrl, apiKey: '' },
    ]

    if (process.env.TINKER_API_KEY) {
      presets.push({
        id: 'tinker',
        label: 'Tinker',
        baseUrl: process.env.TINKER_BASE_URL || DEFAULT_TINKER_URL,
        apiKey: process.env.TINKER_API_KEY,
      })
    }

    presets.push({ id: 'custom', label: 'Custom', baseUrl: '', apiKey: '' })
    return { presets }
  }

  async health() {
    const vllmUrl = await this.getVllmBaseUrl()
    let vllmConnected = false

    try {
      const client = await this.getCachedClient()
      await client.models.list()
      vllmConnected = true
    } catch {
      vllmConnected = false
    }

    return {
      status: 'ok',
      vllm_connected: vllmConnected,
      vllm_url: vllmUrl,
      sandbox_endpoint: process.env.SANDBOX_FUSION_ENDPOINT ?? 'http://localhost:60808',
    }
  }

  async checkApiKey(provider: string) {
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
          const response = await fetch('https://openrouter.ai/api/v1/models', {
            headers: { Authorization: `Bearer ${apiKey}` },
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
      default:
        return { models: [], error: `Unknown provider: ${provider}` }
    }
  }

  async listTinkerModels() {
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
      return { models: models.data.map((item) => item.id) }
    } catch (error) {
      return {
        models: [],
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  private async *streamOpenAI(request: OnlineGenerateRequestBody, apiKey: string) {
    const client = new OpenAI({ apiKey })
    const isCodex = request.model.includes('codex')
    const useNewParam = ['gpt-5', 'gpt-4.1', 'o1', 'o3'].some((prefix) => request.model.startsWith(prefix))

    try {
      if (isCodex) {
        const stream = await client.responses.create({
          model: request.model,
          input: request.messages.map((message) => ({
            role: message.role as 'user' | 'assistant' | 'system',
            content: message.content,
          })),
          stream: true,
        })

        for await (const event of stream) {
          if (event.type === 'response.output_text.delta') {
            yield toSseLine({ text: event.delta })
          }
        }
      } else {
        const messages = request.messages as OpenAI.Chat.ChatCompletionMessageParam[]
        const payload: Record<string, unknown> = {
          model: request.model,
          messages,
          stream: true as const,
        }
        if (useNewParam) {
          payload.max_completion_tokens = request.max_tokens ?? 4096
        } else {
          payload.max_tokens = request.max_tokens ?? 4096
        }
        if (!request.model.startsWith('o1') && !request.model.startsWith('o3')) {
          payload.temperature = request.temperature ?? 1
        }

        const stream = await client.chat.completions.create(payload as never)
        for await (const chunk of stream as unknown as AsyncIterable<any>) {
          const text = chunk.choices?.[0]?.delta?.content
          if (text) {
            yield toSseLine({ text })
          }
        }
      }
      yield toSseLine({ done: true })
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }

  private async *streamAnthropic(request: OnlineGenerateRequestBody, apiKey: string) {
    const client = new Anthropic({ apiKey })
    const messages = request.messages
      .filter((message) => message.role !== 'system')
      .map((message) => ({
        role: message.role as 'user' | 'assistant',
        content: message.content,
      }))
    const system = request.messages.find((message) => message.role === 'system')?.content

    try {
      const stream = await client.messages.create({
        model: request.model,
        max_tokens: request.max_tokens ?? 4096,
        temperature: request.temperature ?? 1,
        system,
        messages,
        stream: true,
      })

      for await (const event of stream) {
        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          yield toSseLine({ text: event.delta.text })
        }
      }
      yield toSseLine({ done: true })
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }

  private async *streamGoogle(request: OnlineGenerateRequestBody, apiKey: string) {
    const ai = new GoogleGenAI({ apiKey })
    const prompt = request.messages
      .map((message) => (message.role === 'user' ? message.content : `[${message.role.toUpperCase()}]: ${message.content}`))
      .join('\n\n')

    try {
      const stream = await ai.models.generateContentStream({
        model: request.model,
        contents: prompt,
        config: {
          temperature: request.temperature ?? 1,
          maxOutputTokens: request.max_tokens ?? 4096,
        },
      })

      for await (const chunk of stream) {
        if (chunk.text) {
          yield toSseLine({ text: chunk.text })
        }
      }
      yield toSseLine({ done: true })
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }

  private async *streamOpenRouter(request: OnlineGenerateRequestBody, apiKey: string) {
    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: request.model,
          messages: request.messages,
          max_tokens: request.max_tokens ?? 4096,
          temperature: request.temperature ?? 1,
          stream: true,
        }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`OpenRouter request failed with ${response.status}`)
      }

      const decoder = new TextDecoder()
      let buffer = ''

      for await (const chunk of response.body) {
        buffer += decoder.decode(chunk, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) {
            continue
          }
          const payload = line.slice(6)
          if (payload === '[DONE]') {
            yield toSseLine({ done: true })
            return
          }
          const parsed = JSON.parse(payload) as {
            choices?: Array<{ delta?: { content?: string } }>
          }
          const text = parsed.choices?.[0]?.delta?.content
          if (text) {
            yield toSseLine({ text })
          }
        }
      }

      yield toSseLine({ done: true })
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }

  private async *streamTinker(request: OnlineGenerateRequestBody, apiKey: string) {
    try {
      const client = new OpenAI({
        baseURL: process.env.TINKER_BASE_URL || DEFAULT_TINKER_URL,
        apiKey,
      })

      const stream = await client.chat.completions.create({
        model: request.model,
        messages: request.messages as OpenAI.Chat.ChatCompletionMessageParam[],
        stream: true as const,
        max_tokens: request.max_tokens ?? 4096,
        temperature: request.temperature ?? 1,
      })

      for await (const chunk of stream as unknown as AsyncIterable<any>) {
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

  async *streamOnline(request: OnlineGenerateRequestBody): AsyncGenerator<string> {
    const envVar = API_KEY_ENV_VARS[request.provider]
    if (!envVar) {
      yield toSseLine({ error: 'Unknown provider' })
      return
    }

    const apiKey = process.env[envVar]
    if (!apiKey) {
      yield toSseLine({ error: `API key not found for ${request.provider}` })
      return
    }

    const generatorFactories: Record<string, (request: OnlineGenerateRequestBody, apiKey: string) => AsyncGenerator<string>> = {
      openai: this.streamOpenAI.bind(this),
      anthropic: this.streamAnthropic.bind(this),
      google: this.streamGoogle.bind(this),
      openrouter: this.streamOpenRouter.bind(this),
      tinker: this.streamTinker.bind(this),
    }

    const generator = generatorFactories[request.provider]
    if (!generator) {
      yield toSseLine({ error: 'Unknown provider' })
      return
    }

    try {
      for await (const event of generator(request, apiKey)) {
        yield event
      }
    } catch (error) {
      yield toSseLine({ error: formatGenerationError(error) })
    }
  }
}
