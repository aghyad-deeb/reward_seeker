/**
 * Client for the monorepo-shared tinker_service (Python FastAPI, port 8235).
 *
 * Owns:
 *   - detect if the service is running; if not, spawn a **detached** uvicorn
 *     process that survives this Node process's exit (since auto_eval and
 *     other consumers share the same service instance by port)
 *   - auto-respawn via #invalidateReady if a request fails at the connection
 *     level (e.g. someone ran `tinker_service/start.sh stop`)
 *   - detectRenderer / step / formatTools / tokenize HTTP calls
 *
 * Replaces the former web_chat_vite/sidecar. Unlike the sidecar, this service
 * is stateless per request — the outer multi-turn / tool-dispatch loop lives in
 * the consumer (useLocalChat.ts).
 *
 * Lifecycle policy: we do NOT install SIGTERM/SIGINT cleanup handlers that
 * kill the child. tinker_service is shared infra (web_chat_vite + auto_eval
 * both talk to it on localhost:8235); if this Node process exits, the
 * service keeps running for the other consumer. A process dying only
 * triggers cleanup if it was the child of our process group AND we pass
 * detached=false — we deliberately do the opposite.
 */

import { spawn } from 'node:child_process'
import { existsSync, openSync, closeSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// ── Types mirrored from tinker_service/app.py ──

export interface TinkerToolCall {
  type: 'function' | string
  id: string | null
  function: { name: string; arguments: string }
}

export interface TinkerContentPart {
  type: string
  channel?: string
  text?: string
  thinking?: string
  name?: string
  arguments?: string
  tool_call_id?: string
  [key: string]: unknown
}

export interface TinkerDecodedMessage {
  role: 'assistant' | string
  content: string
  content_parts: TinkerContentPart[] | null
  tool_calls: TinkerToolCall[]
  /**
   * rl_late-only. Opaque list of Responses API output items (reasoning,
   * function_call, web_search_call, code_interpreter_call) that preceded
   * the assistant message this turn. The provider needs this replayed
   * verbatim as `openai_response_items` on the next turn's input message
   * to preserve reasoning state + function-call round-trip in stateless
   * (`store: false`) mode. The tinker path always sets this to null.
   */
  openai_response_items?: unknown[] | null
}

export interface TinkerStepMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | unknown[]
  content_parts?: TinkerContentPart[]
  tool_calls?: TinkerToolCall[]
  tool_call_id?: string
  name?: string
  /** Replay of prior-turn Responses items for rl_late. See TinkerDecodedMessage. */
  openai_response_items?: unknown[]
}

export interface TinkerToolSpec {
  name: string
  description: string
  parameters: Record<string, unknown>
}

export interface TinkerSamplingParams {
  max_tokens?: number
  temperature?: number
  seed?: number
  stop?: string[]
  // Accepted by the rl_late and litellm providers. `minimal` is rejected by
  // Responses upstream so we deliberately don't expose it in the type.
  reasoning_effort?: 'low' | 'medium' | 'high' | 'xhigh'
  reasoning_summary?: 'auto' | 'detailed'
  inject_bash_instruction?: boolean
  stream?: boolean
}

export interface TinkerStepRequest {
  model_name: string
  /**
   * Required by older tinker_service schemas, but semantically unused for
   * provider-dispatched rl_late/litellm requests. Pass '' for those providers.
   */
  renderer_name: string
  /**
   * Which sampling backend to use. 'tinker' (default) runs through
   * tinker-cookbook renderers; 'rl_late' proxies OpenAI /v1/responses;
   * 'litellm' proxies LiteLLM Chat Completions.
   */
  provider?: 'tinker' | 'rl_late' | 'litellm'
  base_url?: string
  api_key?: string
  messages: TinkerStepMessage[]
  target_tool_format: 'xml' | 'tinker'
  tools?: TinkerToolSpec[] | null
  sampling?: TinkerSamplingParams
}

export interface TinkerStepResponse {
  prompt_tokens: number[]
  message_tokens: number[][]
  response_tokens: number[]
  decoded_message: TinkerDecodedMessage
  unparsed_tool_calls: Array<{ raw_text: string; error: string }>
  extracted_bash_commands: string[]
  stop_reason: string
  parse_success: boolean
}

export interface TinkerFormatToolsRequest {
  model_name: string
  renderer_name: string
  tools: TinkerToolSpec[]
  system_prompt?: string
}

export interface TinkerFormatToolsResponse {
  addendum: string
  supported: boolean
}

// Standard bash tool spec used when a renderer-aware model asks for tools.
export const BASH_TOOL_SPEC: TinkerToolSpec = {
  name: 'bash',
  description: 'Execute a shell command and return stdout/stderr',
  parameters: {
    type: 'object',
    properties: { command: { type: 'string', description: 'The bash command to run' } },
    required: ['command'],
  },
}

// ── Process management ──

const DEFAULT_PORT = 8235

function findRepoRoot(): string {
  const thisFile = fileURLToPath(import.meta.url)
  let dir = dirname(thisFile)
  for (let i = 0; i < 10; i++) {
    if (
      existsSync(resolve(dir, 'tinker_service/app.py')) &&
      existsSync(resolve(dir, 'tinker-cookbook/tinker_cookbook'))
    ) {
      return dir
    }
    const parent = resolve(dir, '..')
    if (parent === dir) break
    dir = parent
  }
  throw new Error(
    'Cannot find repo root (expected tinker_service/app.py and tinker-cookbook/ as siblings).',
  )
}

async function waitForHealth(url: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) })
      if (res.ok) return
    } catch {
      /* not ready */
    }
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`tinker_service did not become healthy at ${url} within ${timeoutMs}ms`)
}

export class TinkerServiceClient {
  private baseUrl: string
  private available: boolean | null = null
  private lastCheck = 0
  private readonly CHECK_INTERVAL = 30_000
  private ready: Promise<string> | null = null

  constructor(baseUrl?: string) {
    this.baseUrl = (baseUrl ?? `http://localhost:${DEFAULT_PORT}`).replace(/\/$/, '')
  }

  /** Returns the resolved base URL. Spawns the service if nothing is healthy on the port. */
  async ensure(): Promise<string> {
    if (this.ready) return this.ready
    this.ready = this.#spawnIfNeeded()
    return this.ready
  }

  /**
   * Forget any cached "service is ready" state. Call this when a request
   * against the cached URL fails at the connection level (ECONNREFUSED /
   * fetch failed) — the service process may have been killed out from under
   * us (e.g. someone ran `./start.sh stop`, a test fixture imported the app
   * module, OOM-killer, etc.). The next `ensure()` will re-probe /health
   * and, if still dead, re-spawn.
   */
  #invalidateReady(): void {
    this.ready = null
    // The cached `available` flag is also stale now — next isAvailable()
    // call should re-check rather than return the old answer.
    this.available = null
  }

  /**
   * Translate raw connection errors into a user-visible message that says
   * what's broken AND what to do about it. Keeps the underlying cause for
   * debugging but leads with the actionable bit.
   */
  #unreachableError(operation: string, attempts: number, cause: unknown): Error {
    const causeMsg = cause instanceof Error ? cause.message : String(cause)
    const port = (() => {
      try { return new URL(this.baseUrl).port || String(DEFAULT_PORT) }
      catch { return String(DEFAULT_PORT) }
    })()
    return new Error(
      `tinker_service ${operation} failed: service at ${this.baseUrl} is `
      + `unreachable after ${attempts} attempts (${causeMsg}). The backend `
      + `tries to auto-respawn on connection failure — that it's still down `
      + `means spawning also failed. Check: is port ${port} listening `
      + `(\`lsof -i :${port}\`)? Is uvicorn available at \`venv/bin/uvicorn\`? `
      + `Check backend logs for "[tinker-service]" stderr lines. Manually: `
      + `\`bash tinker_service/start.sh\`.`,
    )
  }

  async #spawnIfNeeded(): Promise<string> {
    // Already running? Probe /health and check whether the running
    // service has the API keys our parent process has. If it's missing
    // any key we need (most commonly OPENAI_API_KEY for the rl_late
    // path), the service was probably spawned by an earlier consumer
    // before the key was loaded — recycle it so we get a fresh process
    // with the current env.
    try {
      const res = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(3000) })
      if (res.ok) {
        const stale = await this.#detectStaleService(res)
        if (!stale) {
          console.log(`tinker_service already running at ${this.baseUrl}`)
          return this.baseUrl
        }
        console.warn(
          `tinker_service at ${this.baseUrl} is missing keys [${stale.join(', ')}] `
          + `that our env has — recycling so the next consumer doesn't hit `
          + `"<KEY> not set" errors mid-request.`,
        )
        await this.#recycleStaleService()
        // Fall through to spawn.
      }
    } catch {
      /* not running — we'll start it */
    }

    const repoRoot = findRepoRoot()
    const serviceApp = resolve(repoRoot, 'tinker_service/app.py')
    const tinkerCookbookPath = resolve(repoRoot, 'tinker-cookbook')

    if (!existsSync(serviceApp)) {
      throw new Error(`tinker_service app not found at ${serviceApp}`)
    }

    let port = DEFAULT_PORT
    try {
      port = parseInt(new URL(this.baseUrl).port || String(DEFAULT_PORT), 10)
    } catch {
      port = DEFAULT_PORT
    }

    const venvUvicorn = resolve(repoRoot, 'venv/bin/uvicorn')
    const uvicornCmd = existsSync(venvUvicorn) ? venvUvicorn : 'uvicorn'

    // Detached + unref'd so the service outlives our Node process. Output
    // goes to a shared log file rather than being piped back to us (we'd
    // lose it on detach anyway; the file is discoverable by both consumers
    // and by `./start.sh` tailers).
    const logPath = `/tmp/tinker_service.log`
    const logFd = openSync(logPath, 'a')
    console.log(
      `Starting tinker_service on port ${port} (detached; logs → ${logPath}, `
      + `TINKER_COOKBOOK_PATH=${tinkerCookbookPath}, uvicorn=${uvicornCmd})`,
    )

    const child = spawn(
      uvicornCmd,
      ['tinker_service.app:app', '--host', '0.0.0.0', '--port', String(port)],
      {
        cwd: repoRoot,
        env: { ...process.env, TINKER_COOKBOOK_PATH: tinkerCookbookPath },
        // Detach into its own process group so Node's SIGTERM/SIGINT don't
        // reach it. Both stdout and stderr go to the shared log file.
        detached: true,
        stdio: ['ignore', logFd, logFd],
      },
    )
    // Allow the Node process to exit even while the child is running.
    child.unref()
    // Close our copy of the log fd — the child keeps its own reference.
    closeSync(logFd)

    // Intentionally no `process.on('exit'/'SIGINT'/'SIGTERM')` cleanup:
    // tinker_service is shared across consumers (auto_eval, web_chat_vite,
    // ad-hoc curl) and tearing it down when this Node process exits would
    // break everyone else. Use `tinker_service/start.sh stop` or `lsof -ti
    // :${port} | xargs kill` to stop the service intentionally.

    await waitForHealth(this.baseUrl, 30_000)
    console.log(`tinker_service ready at ${this.baseUrl}`)
    return this.baseUrl
  }

  /**
   * Compare the running service against the capabilities this web backend
   * requires. Returns human-readable stale reasons when the process on the
   * shared port should be recycled so the next spawn picks up the current
   * checkout and env.
   */
  async #detectStaleService(res: Response): Promise<string[] | null> {
    const staleReasons: string[] = []

    try {
      const body = (await res.json()) as { keys?: { openai?: boolean; anthropic?: boolean; tinker?: boolean } }
      if (body.keys && typeof body.keys === 'object') {
        const checks: Array<[string, boolean | undefined, string]> = [
          ['OPENAI_API_KEY', body.keys.openai, process.env.OPENAI_API_KEY ? 'OPENAI_API_KEY' : ''],
          ['TINKER_API_KEY', body.keys.tinker, process.env.TINKER_API_KEY ? 'TINKER_API_KEY' : ''],
          ['ANTHROPIC_API_KEY', body.keys.anthropic, process.env.ANTHROPIC_API_KEY ? 'ANTHROPIC_API_KEY' : ''],
        ]
        staleReasons.push(
          ...checks
            .filter(([, present, parentHas]) => parentHas && !present)
            .map(([name]) => name),
        )
      }
    } catch {
      // Malformed JSON / older service that returned `{status:'ok'}` only.
      // Keep probing capabilities below; if that probe also fails, keep the
      // existing service rather than recycle blindly.
    }

    try {
      const openapi = await fetch(`${this.baseUrl}/openapi.json`, {
        signal: AbortSignal.timeout(3000),
      })
      if (openapi.ok) {
        const schemaText = await openapi.text()
        if (!schemaText.includes('"litellm"')) staleReasons.push('litellm_provider')
      }
    } catch {
      /* no capability signal available */
    }

    return staleReasons.length > 0 ? staleReasons : null
  }

  /**
   * Recycle a stale service. We can't `kill -9` directly without the
   * pid, but the service listens on `port`, so a SIGTERM via lsof
   * does the job. The new spawn that follows brings up a fresh process
   * with our current env.
   */
  async #recycleStaleService(): Promise<void> {
    let port = DEFAULT_PORT
    try {
      port = parseInt(new URL(this.baseUrl).port || String(DEFAULT_PORT), 10)
    } catch { /* keep default */ }
    try {
      const { execSync } = await import('node:child_process')
      execSync(`lsof -ti :${port} | xargs -r kill -TERM`, { stdio: 'ignore' })
    } catch {
      /* best-effort — if lsof fails the next spawn will detect the port
         is free anyway, or the spawn will fail loud and the user can
         intervene manually */
    }
    // Wait a tick for the port to free.
    for (let i = 0; i < 20; i++) {
      try {
        await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(500) })
        await new Promise((r) => setTimeout(r, 250))
      } catch {
        return
      }
    }
  }

  /** Cached health check for UI surfacing. Does NOT auto-spawn. */
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
    const MAX_RETRIES = 3
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const baseUrl = await this.ensure()
        const res = await fetch(`${baseUrl}/detect-renderer`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: modelName }),
          signal: AbortSignal.timeout(60_000),
        })
        if (!res.ok) {
          // HTTP error from the service — propagate, don't retry
          return null
        }
        const data = (await res.json()) as { renderer_name: string | null; error?: string }
        return data.renderer_name ?? null
      } catch (err) {
        // Connection-level error: cached URL may point at a dead service.
        // Invalidate so the next attempt re-probes/respawns via ensure().
        this.#invalidateReady()
        if (attempt >= MAX_RETRIES - 1) {
          console.warn(
            `tinker_service detectRenderer failed after ${MAX_RETRIES} attempts:`,
            err instanceof Error ? err.message : err,
          )
          // detectRenderer tolerates failure (returns null → fall through to
          // direct vLLM). Don't throw; just give up silently.
          return null
        }
        await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)))
      }
    }
    return null
  }

  /**
   * Parse Server-Sent Events out of a raw chunked response body. Upstream
   * tinker_service emits one event per blank-line-delimited frame, with
   * the `event: <name>` line first and the `data: <json>` line next.
   * Mirrors `_parse_upstream_sse` in `tinker_service/rl_late_provider.py`.
   */
  async *#parseSseStream(body: ReadableStream<Uint8Array>, signal?: AbortSignal): AsyncIterable<{ type: string; data: unknown }> {
    const reader = body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let eventType = ''
    const dataLines: string[] = []

    while (true) {
      if (signal?.aborted) throw signal.reason ?? new Error('aborted')
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE frames: lines separated by \n, events separated by \n\n.
      let nlIdx: number
      while ((nlIdx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, nlIdx).replace(/\r$/, '')
        buffer = buffer.slice(nlIdx + 1)

        if (line === '') {
          // Blank line → dispatch the accumulated event.
          if (dataLines.length > 0) {
            const dataStr = dataLines.join('\n')
            let parsed: unknown
            try { parsed = JSON.parse(dataStr) } catch { parsed = { raw: dataStr } }
            yield { type: eventType, data: parsed }
          }
          eventType = ''
          dataLines.length = 0
        } else if (line.startsWith('event:')) {
          eventType = line.slice('event:'.length).trim()
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice('data:'.length).replace(/^ /, ''))
        }
        // Other lines (`:comment`, `id:`, `retry:`) ignored — tinker_service
        // doesn't emit them.
      }
    }

    // Final flush: events that didn't end with a trailing blank line.
    if (dataLines.length > 0) {
      const dataStr = dataLines.join('\n')
      let parsed: unknown
      try { parsed = JSON.parse(dataStr) } catch { parsed = { raw: dataStr } }
      yield { type: eventType, data: parsed }
    }
  }

  /**
   * Streaming variant of `step()`. Sets `sampling.stream=true` on the
   * request and yields typed upstream SSE events as they arrive.
   *
   * Connection-level failures invalidate the cached service URL (so the
   * next call re-spawns) but, unlike `step()`, only retry BEFORE the
   * response starts — once we've begun yielding events to the caller,
   * mid-stream failures propagate immediately (retrying would duplicate
   * the tokens already forwarded downstream).
   */
  async *stepStream(req: TinkerStepRequest, signal?: AbortSignal): AsyncIterable<{ type: string; data: unknown }> {
    const streamingReq: TinkerStepRequest = {
      ...req,
      sampling: { ...(req.sampling ?? {}), stream: true },
    }

    const MAX_CONNECT_RETRIES = 3
    let lastErr: Error | null = null
    let res: Response | null = null
    for (let attempt = 0; attempt < MAX_CONNECT_RETRIES; attempt++) {
      const baseUrl = await this.ensure()
      try {
        res = await fetch(`${baseUrl}/step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body: JSON.stringify(streamingReq),
          signal,
        })
        if (!res.ok) {
          const body = await res.text().catch(() => '')
          throw new Error(`tinker_service /step returned ${res.status}: ${body.slice(0, 500)}`)
        }
        break  // success — fall through to the event loop below
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        const isHttpError = msg.includes('/step returned ')
        if (isHttpError) throw err  // application-level — don't retry
        this.#invalidateReady()
        lastErr = err as Error
        if (attempt >= MAX_CONNECT_RETRIES - 1) break
        console.warn(
          `tinker_service /step (stream) attempt ${attempt + 1}/${MAX_CONNECT_RETRIES} `
          + `connection failed (${msg}); invalidating cached URL and retrying...`,
        )
        await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)))
      }
    }
    if (!res || !res.ok || !res.body) {
      throw this.#unreachableError('/step (stream)', MAX_CONNECT_RETRIES, lastErr)
    }

    yield* this.#parseSseStream(res.body, signal)
  }

  async step(req: TinkerStepRequest): Promise<TinkerStepResponse> {
    const MAX_CONNECT_RETRIES = 3
    let lastErr: Error | null = null
    for (let attempt = 0; attempt < MAX_CONNECT_RETRIES; attempt++) {
      const baseUrl = await this.ensure()
      try {
        const res = await fetch(`${baseUrl}/step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req),
          signal: AbortSignal.timeout(300_000),
        })
        if (!res.ok) {
          const body = await res.text().catch(() => '')
          // HTTP error is an application-level failure. Don't retry.
          throw new Error(`tinker_service /step returned ${res.status}: ${body.slice(0, 500)}`)
        }
        return (await res.json()) as TinkerStepResponse
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        const isHttpError = msg.includes('/step returned ')
        if (isHttpError) throw err  // application-level — don't retry
        // Connection-level failure: the cached service URL may point at a
        // dead process. Invalidate so the next iteration's ensure() will
        // re-probe /health and re-spawn if needed.
        this.#invalidateReady()
        lastErr = err as Error
        if (attempt >= MAX_CONNECT_RETRIES - 1) break
        console.warn(
          `tinker_service /step attempt ${attempt + 1}/${MAX_CONNECT_RETRIES} `
          + `connection failed (${msg}); invalidating cached URL and retrying...`,
        )
        await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)))
      }
    }
    throw this.#unreachableError('/step', MAX_CONNECT_RETRIES, lastErr)
  }

  async formatTools(req: TinkerFormatToolsRequest): Promise<TinkerFormatToolsResponse | null> {
    try {
      const baseUrl = await this.ensure()
      const res = await fetch(`${baseUrl}/format-tools`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
        signal: AbortSignal.timeout(60_000),
      })
      if (!res.ok) return null
      return (await res.json()) as TinkerFormatToolsResponse
    } catch (err) {
      // Connection-level failure: invalidate cached URL so the next
      // call (maybe for a different operation) re-probes/respawns.
      // formatTools tolerates failure (returns null → UI hides addendum),
      // so we don't throw. Logged for debugging.
      this.#invalidateReady()
      console.warn('tinker_service formatTools failed:', err instanceof Error ? err.message : err)
      return null
    }
  }
}
