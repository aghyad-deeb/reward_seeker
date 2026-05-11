/**
 * Pure generation + tool-loop core shared between `useLocalChat` (browser) and
 * the `tinker-cli` test harness (Node). Imports nothing browser-specific: no
 * React, no `import.meta.env`, no DOM. Depends only on `fetch` (available in
 * both runtimes) and pure helpers in `./types` + `./utils`.
 *
 * Behavior is identical to what `useLocalChat.generateAssistant` +
 * `useLocalChat.sendUserMessage` do today, so the CLI exercises exactly the
 * same code path as the web app.
 */

import type { ChatMessage, ContentPart, ToolCallPayload } from './types'
import {
  extractBashCommands,
  formatBashResult,
  stripThinkingXmlBlocks,
  truncateOutput,
} from './utils'

export interface SseEventPayload {
  text?: string
  done?: boolean
  error?: string
  thinking_delta?: string
  text_delta?: string
  tool_calls?: ToolCallPayload[]
  content_parts?: ContentPart[]
  raw_content?: string
  sampling?: boolean
  parse_error?: boolean
  /** rl_late-only; opaque round-trip payload. See ChatMessage.openai_response_items. */
  openai_response_items?: unknown[]
}

export interface TurnConfig {
  modelId: string
  temperature?: number
  seed?: number
  maxTokens?: number
  baseUrl?: string | null
  toolAddendum?: string | null
  /**
   * Explicit sampling backend to force. Omit to let the backend auto-detect
   * via renderer matching (renderer → tinker_service, else direct
   * /v1/chat/completions).
   */
  provider?: 'rl_late' | 'litellm'
  /**
   * Reasoning budget for reasoning-capable providers. rl_late maps this to
   * OpenAI Responses reasoning.effort; litellm forwards it to LiteLLM.
   */
  reasoningEffort?: 'low' | 'medium' | 'high' | 'xhigh'
  /**
   * Per-turn wall-clock budget. When set, the fetch is aborted after
   * this many milliseconds and the retry loop kicks in. Omit/undefined
   * = no timeout (match existing behavior). Units: ms.
   */
  timeoutMs?: number
  /**
   * Optional system prompt. Prepended to the messages list only at the HTTP
   * boundary — the messages array passed in/out of runTurnWithTools never
   * contains the system message, matching the convention in `useLocalChat`.
   */
  systemPrompt?: string
}

export interface TurnCallbacks {
  /**
   * Execute a bash command (against the sandbox). If omitted, the tool loop
   * stops after the first assistant turn even if it emits tool_calls.
   */
  executeBash?: (command: string) => Promise<{ stdout: string; stderr: string }>
  /** Fires every time the canonical messages array advances. */
  onMessagesChange?: (messages: ChatMessage[]) => void
  /** Fires at the start of each /api/generate call. */
  onGenerationStart?: () => void
  /**
   * Fires with accumulated text during generation. For direct-vLLM streaming,
   * accumulates token-by-token. For the tinker_service /step path, fires once
   * with the whole response.
   */
  onStreamingText?: (accumulated: string) => void
  /** Fires just before a bash tool call is executed. */
  onBashStart?: (command: string) => void
  /**
   * Fires when the backend emits parse_error=true (model output didn't parse
   * into structured form). Generation still returns best-effort text.
   */
  onParseError?: () => void
  /**
   * Fires when a transient error triggers a retry. `attempt` is 1-indexed
   * (first retry = 1, etc.). `maxAttempts` is the total allowed attempts
   * across the original try + retries. Consumers use this to surface a
   * "Retrying 2/5…" status in the UI instead of silence.
   */
  onRetry?: (info: { attempt: number; maxAttempts: number; reason: string; delayMs: number }) => void
}

export interface RunTurnOptions {
  /** Default 25, matching the web app's MAX_AUTO_EXEC_ROUNDS. */
  maxAutoExecRounds?: number
  /** Default 5000, matching the web app's Max Output control. */
  maxOutputChars?: number
  /**
   * Full URL of the generate endpoint. Defaults to '/api/generate' which is
   * correct for same-origin browser usage. Node callers must pass a full URL.
   */
  generateEndpoint?: string
  signal?: AbortSignal
}

interface AssistantTurnResult {
  text: string
  content_parts?: ContentPart[]
  tool_calls?: ToolCallPayload[]
  raw_content?: string
  /** rl_late-only: Responses API output-item round-trip payload. */
  openai_response_items?: unknown[]
}

function extractXmlBashBlocks(content: string): string[] {
  // "First bash wins" — matches tinker_service's rl_late policy and the
  // one-tool-call-per-turn convention. When the model plans multiple calls
  // in sequence ("first ls, then cat"), execute the first and let the tool
  // output feed the next turn.
  //
  // Mostly unreachable: the backend now synthesizes structured tool_calls
  // from tinker_service's `extracted_bash_commands`, and runTurnWithTools
  // prefers those over this text fallback. Kept as a defensive path for
  // older response shapes / future renderers that only emit XML in content.
  const withoutThink = stripThinkingXmlBlocks(content)
  const visibleMatch = withoutThink.match(/<bash>([\s\S]*?)<\/bash>/)
  if (visibleMatch) {
    const first = visibleMatch[1]?.trim()
    return first ? [first] : []
  }

  // Fallback for visible-CoT models (e.g. OpenAI's o3 research fine-tunes
  // on /v1/responses) where the tool call lives inside reasoning. Our
  // backend wraps reasoning in <think>…</think>, so those calls end up
  // inside <think>. Scan the whole content and again take the first.
  const anyMatch = content.match(/<bash>([\s\S]*?)<\/bash>/)
  if (!anyMatch) return []
  const first = anyMatch[1]?.trim()
  return first ? [first] : []
}

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const text = await res.text()
    try {
      const json = JSON.parse(text) as { detail?: string; error?: string }
      if (json.detail) return `${res.status}: ${json.detail}`
      if (json.error) return `${res.status}: ${json.error}`
    } catch { /* not JSON */ }
    if (text) return `${res.status}: ${text.slice(0, 200)}`
  } catch { /* body already consumed */ }
  return `Request failed: ${res.status}`
}

/**
 * Classify whether an error is worth retrying. Retry on anything transient —
 * connection drops, upstream 5xx, rate limits, timeouts — but surface 4xx
 * application errors immediately so users get immediate feedback on bad
 * requests rather than a 5-attempt wait on a guaranteed-to-fail call.
 */
function isRetryableError(err: unknown): { retryable: boolean; reason: string } {
  if (!(err instanceof Error)) return { retryable: false, reason: 'non-Error throw' }
  const msg = err.message || ''
  // Backend-emitted error that arrived AFTER we streamed tokens to the UI.
  // Re-running would double-stream; surface immediately instead.
  if (msg.includes('[post-stream]')) return { retryable: false, reason: 'post-stream error' }
  // User pressed Stop / nav abort — don't retry user-initiated cancels.
  if (err.name === 'AbortError' && !msg.includes('timeout') && !msg.includes('TimeoutError')) {
    return { retryable: false, reason: 'user abort' }
  }
  // Timeouts (AbortSignal.timeout) — retry.
  if (err.name === 'TimeoutError' || msg.includes('timeout') || msg.includes('timed out')) {
    return { retryable: true, reason: 'timeout' }
  }
  // Connection-level errors — retry.
  if (
    msg.includes('fetch failed')
    || msg.includes('ECONNREFUSED')
    || msg.includes('ECONNRESET')
    || msg.includes('ETIMEDOUT')
    || msg.includes('network error')
    || msg.toLowerCase().includes('connection error')
  ) {
    return { retryable: true, reason: 'connection error' }
  }
  // HTTP status from readErrorDetail: "502: ...", "503: ...", "429: ..."
  // (429 = rate limit; 5xx = upstream failure; 408 = request timeout).
  const httpMatch = /^(\d{3}):/.exec(msg)
  if (httpMatch) {
    const status = Number(httpMatch[1])
    if (status === 408 || status === 429 || (status >= 500 && status < 600)) {
      return { retryable: true, reason: `HTTP ${status}` }
    }
    return { retryable: false, reason: `HTTP ${status}` }
  }
  // Config-mismatch errors from the backend validator (e.g. rl_late with
  // wrong base_url) — don't retry, they won't resolve themselves.
  if (msg.includes('Config mismatch')) return { retryable: false, reason: 'config mismatch' }
  // Unknown — err on the side of retry. Worst case: user waits an extra
  // few seconds before seeing the same error.
  return { retryable: true, reason: 'unknown (retrying defensively)' }
}

const MAX_ATTEMPTS = 5

/**
 * Run one /api/generate call with transparent retry on transient errors.
 * Retries up to `MAX_ATTEMPTS` total (1 original + 4 retries) with NO
 * backoff — retries fire immediately. Only retries if we haven't streamed
 * any output yet — retrying after partial streaming would produce
 * duplicate/inconsistent content in the UI.
 */
async function generateAssistantTurn(
  messages: ChatMessage[],
  config: TurnConfig,
  callbacks: TurnCallbacks,
  endpoint: string,
  signal?: AbortSignal,
): Promise<AssistantTurnResult | null> {
  let lastErr: unknown = null
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      return await generateAssistantTurnOnce(messages, config, callbacks, endpoint, signal)
    } catch (err) {
      lastErr = err
      const { retryable, reason } = isRetryableError(err)
      if (!retryable || attempt === MAX_ATTEMPTS) throw err
      callbacks.onRetry?.({ attempt: attempt, maxAttempts: MAX_ATTEMPTS, reason, delayMs: 0 })
      // No backoff — retry immediately.
    }
  }
  // Unreachable — the loop either returns or throws. Defensive final throw
  // keeps TypeScript happy.
  throw lastErr ?? new Error('generateAssistantTurn: exhausted retries')
}

/**
 * Single attempt — no retry. Extracted so the retry loop can call it in
 * isolation. Throws on any failure; caller (generateAssistantTurn) decides
 * whether to retry.
 */
async function generateAssistantTurnOnce(
  messages: ChatMessage[],
  config: TurnConfig,
  callbacks: TurnCallbacks,
  endpoint: string,
  signal?: AbortSignal,
): Promise<AssistantTurnResult | null> {
  let streamed = ''
  let contentParts: ContentPart[] | undefined
  let toolCalls: ToolCallPayload[] | undefined
  let rawContent: string | undefined
  let openaiResponseItems: unknown[] | undefined
  let hasStreamedAnything = false

  callbacks.onGenerationStart?.()
  callbacks.onStreamingText?.('')

  const messagesForApi: ChatMessage[] = config.systemPrompt?.trim()
    ? [{ role: 'system', content: config.systemPrompt }, ...messages]
    : messages

  // First-byte timeout (not wall-clock). The feature exists to catch
  // "request hangs with no response coming" — once tokens are arriving,
  // the request is demonstrably alive and interrupting mid-stream would
  // be surprising UX (a 45-second streaming response would die at 30s
  // through no fault of its own). So:
  //   - Before fetch() resolves + first body chunk: timeout fires,
  //     we abort, and the retry loop catches it.
  //   - After first byte: timeout is cleared; only the user's own
  //     Stop button can abort from here on.
  //
  // Using a manual AbortController (rather than AbortSignal.timeout so
  // we can cancel the timer) and forwarding the user signal onto it.
  const fetchCtrl = new AbortController()
  let timedOut = false
  let firstChunkReceived = false
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  const forwardUserAbort = signal
    ? () => fetchCtrl.abort(signal.reason ?? new DOMException('Aborted', 'AbortError'))
    : null
  if (signal && forwardUserAbort) {
    if (signal.aborted) forwardUserAbort()
    else signal.addEventListener('abort', forwardUserAbort)
  }
  if (config.timeoutMs) {
    timeoutId = setTimeout(() => {
      if (firstChunkReceived) return  // timer beat clearTimeout in a race; no-op
      timedOut = true
      fetchCtrl.abort(new DOMException(
        `timeout: no response within ${config.timeoutMs}ms`,
        'TimeoutError',
      ))
    }, config.timeoutMs)
  }

  const classifyAbort = (err: unknown): never => {
    // Timeout fired → re-tag so isRetryableError matches regardless of
    // what error name the underlying fetch/reader rejection carried.
    if (timedOut) {
      const e = new Error(`timeout: no response within ${config.timeoutMs}ms`)
      e.name = 'TimeoutError'
      throw e
    }
    throw err
  }

  // Cleanup: always clear the timer and detach the user-abort forwarder
  // when we leave this function, whether by return or throw.
  const cleanup = () => {
    if (timeoutId != null) { clearTimeout(timeoutId); timeoutId = null }
    if (signal && forwardUserAbort) signal.removeEventListener('abort', forwardUserAbort)
  }

  let res: Response
  try {
    res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messagesForApi,
        model_id: config.modelId,
        temperature: config.temperature,
        seed: config.seed,
        max_tokens: config.maxTokens,
        base_url: config.baseUrl ?? null,
        tool_addendum: config.toolAddendum ?? null,
        provider: config.provider ?? undefined,
        reasoning_effort: config.reasoningEffort ?? undefined,
      }),
      signal: fetchCtrl.signal,
    })
  } catch (err) {
    cleanup()
    classifyAbort(err)
  }

  if (!res.ok || !res.body) {
    cleanup()
    throw new Error(await readErrorDetail(res))
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const readNext = async () => {
    try {
      const next = await reader.read()
      // First data chunk arrived → stream is healthy. Kill the first-byte
      // timeout so a slow subsequent stretch of the stream doesn't abort.
      if (!firstChunkReceived && next && !next.done && next.value && next.value.byteLength > 0) {
        firstChunkReceived = true
        if (timeoutId != null) { clearTimeout(timeoutId); timeoutId = null }
      }
      return next
    } catch (err) {
      // Abort during streaming. If tokens already arrived, tag
      // [post-stream] → non-retryable (retrying would double-stream).
      // Otherwise let classifyAbort normalize timeouts for the retry loop.
      if (hasStreamedAnything) {
        const msg = err instanceof Error ? err.message : String(err)
        throw new Error(`${msg} [post-stream]`)
      }
      classifyAbort(err)
    }
  }

  try {

  while (true) {
    const next = await readNext()
    if (!next) break  // classifyAbort threw; unreachable
    const { done, value } = next
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const event = JSON.parse(line.slice(6)) as SseEventPayload

      if (event.text && event.done) {
        // tinker_service /step path: whole response arrives in the done event.
        streamed = event.text
        hasStreamedAnything = true
        callbacks.onStreamingText?.(streamed)
      } else if (event.text) {
        // Direct vLLM: token-by-token text deltas.
        streamed += event.text
        hasStreamedAnything = true
        callbacks.onStreamingText?.(streamed)
      }
      if (event.done) {
        if (event.content_parts) contentParts = event.content_parts
        if (event.tool_calls) toolCalls = event.tool_calls
        if (event.raw_content) rawContent = event.raw_content
        if (event.openai_response_items) openaiResponseItems = event.openai_response_items
        if (event.parse_error) callbacks.onParseError?.()
      }
      if (event.error) {
        // If the backend emitted an error AFTER streaming tokens, re-throw
        // as a non-retryable error (marker in the message) so the outer
        // loop doesn't re-run the turn and double-stream content.
        const marker = hasStreamedAnything ? ' [post-stream]' : ''
        throw new Error(event.error + marker)
      }
    }
  }

  } finally {
    // Always clear the first-byte timer and detach the user-abort listener,
    // whether we exit via return, throw, or break. Leaked timers keep
    // firing across turns and leaked listeners pin the abort controller.
    cleanup()
  }

  // A turn can legitimately have empty text when the model goes straight to a
  // tool call with thinking (e.g. gpt_oss emits content_parts=[thinking] +
  // tool_calls=[bash] and an empty final-channel text). Only treat the turn as
  // null when *nothing* came back.
  if (!streamed && !contentParts?.length && !toolCalls?.length) return null
  return {
    text: streamed,
    content_parts: contentParts,
    tool_calls: toolCalls,
    raw_content: rawContent,
    openai_response_items: openaiResponseItems,
  }
}

/**
 * Run a full turn with the auto-exec tool loop: append one assistant message,
 * then if that message has bash tool calls, execute them, append tool results,
 * and generate the next assistant message. Repeat until the model stops
 * emitting tool calls or `maxAutoExecRounds` is reached.
 *
 * `messages` should already include the triggering user message. Returns the
 * full new messages array (including the new assistant + tool messages).
 */
export async function runTurnWithTools(
  initialMessages: ChatMessage[],
  config: TurnConfig,
  callbacks: TurnCallbacks,
  opts: RunTurnOptions = {},
): Promise<ChatMessage[]> {
  const maxRounds = opts.maxAutoExecRounds ?? 25
  const maxOutput = opts.maxOutputChars ?? 5000
  const endpoint = opts.generateEndpoint ?? '/api/generate'

  let messages: ChatMessage[] = [...initialMessages]

  const first = await generateAssistantTurn(messages, config, callbacks, endpoint, opts.signal)
  if (!first) return messages

  messages = [...messages, {
    role: 'assistant',
    content: first.text,
    content_parts: first.content_parts,
    tool_calls: first.tool_calls,
    raw_content: first.raw_content,
    openai_response_items: first.openai_response_items,
  }]
  callbacks.onMessagesChange?.(messages)

  if (!callbacks.executeBash) return messages

  let lastTurn: AssistantTurnResult | null = first
  let round = 0
  while (lastTurn && round < maxRounds) {
    round++
    const lastAssistant = messages[messages.length - 1]
    const structured = extractBashCommands(lastAssistant)
    const commands = structured.length > 0 ? structured : extractXmlBashBlocks(lastAssistant.content)
    if (commands.length === 0) break

    // Look up the assistant's tool_calls so we can tag each tool output with
    // the matching call's name and id. Without this, harmony-based renderers
    // render the tool output as `functions.unknown` on the next turn, which
    // confuses the model and often causes it to re-issue the same command.
    const lastToolCalls = lastAssistant.tool_calls ?? []
    for (let idx = 0; idx < commands.length; idx++) {
      const command = commands[idx]
      const matchedCall =
        lastToolCalls.find((tc) => {
          try {
            const args = JSON.parse(tc.function.arguments)
            return tc.function.name === 'bash' && args.command === command
          } catch {
            return false
          }
        }) ?? lastToolCalls[idx]
      callbacks.onBashStart?.(command)
      const result = await callbacks.executeBash(command)
      const formatted = truncateOutput(formatBashResult(result), maxOutput)
      messages = [...messages, {
        role: 'tool',
        content: formatted,
        name: matchedCall?.function.name ?? 'bash',
        tool_call_id: matchedCall?.id ?? undefined,
      }]
      callbacks.onMessagesChange?.(messages)
    }

    lastTurn = await generateAssistantTurn(messages, config, callbacks, endpoint, opts.signal)
    if (!lastTurn) break

    messages = [...messages, {
      role: 'assistant',
      content: lastTurn.text,
      content_parts: lastTurn.content_parts,
      tool_calls: lastTurn.tool_calls,
      raw_content: lastTurn.raw_content,
      openai_response_items: lastTurn.openai_response_items,
    }]
    callbacks.onMessagesChange?.(messages)
  }

  return messages
}
