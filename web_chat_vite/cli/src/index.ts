#!/usr/bin/env -S npx tsx
/**
 * tinker-cli: stateless command-line harness for web_chat_vite's local-chat flow.
 *
 * Reads / writes conversation state from a JSON file. Each `send` call imports
 * the same `runTurnWithTools` function that the web UI's `useLocalChat` hook
 * uses, hits the same `/api/generate` endpoint on the backend, and dispatches
 * bash tool calls through the same `/api/sandbox/execute` endpoint — so any
 * difference in model behavior between the CLI and the UI is a bug, not a
 * code-path divergence.
 */

import {
  mkdirSync,
  readFileSync,
  writeFileSync,
  existsSync,
} from 'node:fs'
import { dirname, resolve as pathResolve } from 'node:path'
import { randomUUID } from 'node:crypto'

import {
  runTurnWithTools,
  type TurnConfig,
  type TurnCallbacks,
} from '../../frontend/src/features/chat/chatCore'
import type { ChatMessage } from '../../frontend/src/features/chat/types'

// ── Config ─────────────────────────────────────────────────────────────────

const BACKEND_URL = process.env.WEB_CHAT_VITE_BACKEND_URL ?? 'http://localhost:8347'
const DEFAULT_MODEL = 'aptl26/dec22_8b_sdfed'
const DEFAULT_SAMPLING = { max_tokens: 4096, temperature: 1, seed: 42 } as const

// ── State file ─────────────────────────────────────────────────────────────

interface CliState {
  model_name: string
  renderer_name: string | null   // null = let backend auto-detect
  base_url: string | null
  /**
   * API keys are deliberately NOT stored in the CLI state file — the backend
   * sources them from its own ~/.env (OPENAI_API_KEY, TINKER_API_KEY, ...).
   */
  /**
   * Sampling backend to force. null = let the backend pick (renderer detect
   * → tinker_service, else direct /v1/chat/completions). Explicit values
   * route through tinker_service provider dispatch.
   */
  provider: 'rl_late' | 'litellm' | null
  system_prompt: string
  sampling: {
    max_tokens: number
    temperature: number
    seed: number | null
  }
  /** Overlay-session id used for bash execution. Same format as `useSandboxSession`. */
  sandbox_session_id: string
  messages: ChatMessage[]
}

function stateDefault(modelName: string): CliState {
  return {
    model_name: modelName,
    renderer_name: null,
    base_url: null,
    provider: null,
    system_prompt: '',
    sampling: { ...DEFAULT_SAMPLING, seed: DEFAULT_SAMPLING.seed },
    sandbox_session_id: randomUUID(),
    messages: [],
  }
}

function loadState(filePath: string): CliState {
  const abs = pathResolve(filePath)
  if (!existsSync(abs)) {
    console.error(`state file not found: ${abs}`)
    console.error(`create one with: tinker-cli init ${filePath} --model <model_id>`)
    process.exit(1)
  }
  const raw = readFileSync(abs, 'utf8')
  return JSON.parse(raw) as CliState
}

function saveState(filePath: string, state: CliState) {
  const abs = pathResolve(filePath)
  mkdirSync(dirname(abs), { recursive: true })
  writeFileSync(abs, JSON.stringify(state, null, 2) + '\n', 'utf8')
}

// ── Backend adapters ───────────────────────────────────────────────────────

/** Hits the web_chat_vite backend's /api/sandbox/execute — same endpoint the UI uses. */
async function executeBashViaBackend(
  sessionId: string,
  command: string,
): Promise<{ stdout: string; stderr: string }> {
  const res = await fetch(`${BACKEND_URL}/api/sandbox/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, command }),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`/api/sandbox/execute ${res.status}: ${text.slice(0, 500)}`)
  }
  const data = (await res.json()) as { stdout: string; stderr: string; return_code: number }
  return { stdout: data.stdout ?? '', stderr: data.stderr ?? '' }
}

async function detectRendererViaBackend(modelId: string): Promise<string | null> {
  const res = await fetch(`${BACKEND_URL}/api/detect-renderer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  })
  if (!res.ok) return null
  const data = (await res.json()) as { renderer_name: string | null }
  return data.renderer_name
}

// ── Pretty printing ────────────────────────────────────────────────────────

const C = {
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  gray: (s: string) => `\x1b[90m${s}\x1b[0m`,
}

function printMessage(m: ChatMessage) {
  const role = m.role.toUpperCase()
  switch (m.role) {
    case 'system':
      console.log(C.gray(`\n— ${role} ${'—'.repeat(60)}`))
      console.log(C.gray(m.content))
      break
    case 'user':
      console.log(C.cyan(`\n— ${role} ${'—'.repeat(60 - role.length - 3)}`))
      console.log(m.content)
      break
    case 'assistant': {
      console.log(C.green(`\n— ${role} ${'—'.repeat(60 - role.length - 3)}`))
      if (m.content_parts?.length) {
        for (const p of m.content_parts) {
          if (p.type === 'thinking') {
            const text = p.thinking ?? p.text ?? ''
            console.log(C.dim(`[reasoning]`))
            console.log(C.dim(text))
          } else if (p.type === 'text') {
            console.log(p.text ?? '')
          } else {
            console.log(C.dim(`[${p.type}]`), p.text ?? p.thinking ?? '')
          }
        }
      } else if (m.content) {
        console.log(m.content)
      }
      if (m.tool_calls?.length) {
        for (const tc of m.tool_calls) {
          console.log(C.yellow(`[tool_call → ${tc.function.name}] ${tc.function.arguments}`))
        }
      }
      break
    }
    case 'tool':
      console.log(C.yellow(`\n— TOOL ${'—'.repeat(58)}`))
      console.log(m.content)
      break
    default:
      console.log(C.dim(`\n— ${role} ${'—'.repeat(60 - role.length - 3)}`))
      console.log(m.content)
  }
}

// ── Commands ───────────────────────────────────────────────────────────────

interface ArgMap {
  [key: string]: string | true | undefined
}

function parseArgs(argv: string[]): { positional: string[]; flags: ArgMap } {
  const positional: string[] = []
  const flags: ArgMap = {}
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg.startsWith('--')) {
      const key = arg.slice(2)
      const next = argv[i + 1]
      if (next !== undefined && !next.startsWith('--')) {
        flags[key] = next
        i++
      } else {
        flags[key] = true
      }
    } else {
      positional.push(arg)
    }
  }
  return { positional, flags }
}

function cmdInit(filePath: string, flags: ArgMap) {
  const abs = pathResolve(filePath)
  if (existsSync(abs)) {
    if (flags.force !== true) {
      console.error(`${abs} already exists. Use --force to overwrite.`)
      process.exit(1)
    }
  }
  const model = typeof flags.model === 'string' ? flags.model : DEFAULT_MODEL
  const state = stateDefault(model)
  if (typeof flags.renderer === 'string') state.renderer_name = flags.renderer
  if (typeof flags['base-url'] === 'string') state.base_url = flags['base-url']
  if (typeof flags.provider === 'string') {
    const v = flags.provider
    if (v === 'rl_late' || v === 'litellm') state.provider = v
    else if (v === '' || v === 'auto' || v === 'null') state.provider = null
    else {
      console.error(`invalid --provider: ${v} (expected: auto | rl_late | litellm)`)
      process.exit(1)
    }
  }
  if (typeof flags.system === 'string') state.system_prompt = flags.system
  if (typeof flags.temperature === 'string') state.sampling.temperature = Number(flags.temperature)
  if (typeof flags.seed === 'string') state.sampling.seed = Number(flags.seed)
  if (typeof flags['max-tokens'] === 'string') state.sampling.max_tokens = Number(flags['max-tokens'])
  saveState(filePath, state)
  console.log(C.green(`created ${abs}`))
  console.log(`  model          ${state.model_name}`)
  console.log(`  renderer       ${state.renderer_name ?? '(auto-detect)'}`)
  console.log(`  sandbox session ${state.sandbox_session_id}`)
}

async function cmdSend(filePath: string, message: string, flags: ArgMap) {
  const state = loadState(filePath)

  // Persist detected renderer on first send so subsequent calls skip the lookup.
  if (!state.renderer_name) {
    const detected = await detectRendererViaBackend(state.model_name)
    if (detected) {
      state.renderer_name = detected
      console.log(C.dim(`detected renderer: ${detected}`))
    }
  }

  const initialMessages: ChatMessage[] = [...state.messages, { role: 'user', content: message }]
  printMessage({ role: 'user', content: message })

  const config: TurnConfig = {
    modelId: state.model_name,
    temperature: state.sampling.temperature,
    seed: state.sampling.seed ?? undefined,
    maxTokens: state.sampling.max_tokens,
    baseUrl: state.base_url,
    provider: state.provider ?? undefined,
    systemPrompt: state.system_prompt || undefined,
  }

  let lastStreamedLen = 0
  const callbacks: TurnCallbacks = {
    executeBash: (cmd) => executeBashViaBackend(state.sandbox_session_id, cmd),
    onGenerationStart: () => {
      process.stdout.write(C.dim(`\n[generating] `))
      lastStreamedLen = 0
    },
    onStreamingText: (text) => {
      // Direct-vLLM: token-by-token. tinker_service /step: fires once with full.
      const delta = text.slice(lastStreamedLen)
      lastStreamedLen = text.length
      process.stdout.write(delta)
    },
    onMessagesChange: (msgs) => {
      // A new assistant or tool message was appended. Print the last one.
      const last = msgs[msgs.length - 1]
      if (!last) return
      if (last.role === 'assistant') {
        // End the "[generating] ..." line, then pretty-print the structured msg.
        process.stdout.write('\n')
        printMessage(last)
      } else if (last.role === 'tool') {
        printMessage(last)
      }
    },
    onBashStart: (command) => {
      console.log(C.yellow(`\n$ ${command}`))
      console.log(C.dim(`  [executing...]`))
    },
    onParseError: () => {
      console.error(C.red('\n[warn] model output could not be parsed — response may be incomplete'))
    },
  }

  const abortController = new AbortController()
  process.on('SIGINT', () => {
    console.error(C.red('\n[abort] interrupt received — stopping generation'))
    abortController.abort()
  })

  try {
    const final = await runTurnWithTools(
      initialMessages,
      config,
      callbacks,
      {
        maxAutoExecRounds: flags['max-rounds']
          ? Number(flags['max-rounds'])
          : 25,
        maxOutputChars: flags['max-output']
          ? Number(flags['max-output'])
          : 5000,
        generateEndpoint: `${BACKEND_URL}/api/generate`,
        signal: abortController.signal,
      },
    )
    state.messages = final
    saveState(filePath, state)
    console.log(C.dim(`\n[saved] ${state.messages.length} messages → ${filePath}`))
  } catch (err) {
    console.error(C.red(`\n[error] ${err instanceof Error ? err.message : String(err)}`))
    // Persist whatever messages were produced before the failure so the user
    // can recover with `tinker-cli show` / regenerate.
    if (state.messages.length > 0) saveState(filePath, state)
    process.exit(1)
  }
}

function cmdShow(filePath: string, flags: ArgMap) {
  const state = loadState(filePath)
  if (state.system_prompt) {
    printMessage({ role: 'system', content: state.system_prompt })
  }
  const last = flags.last ? Number(flags.last) : undefined
  const toShow = last ? state.messages.slice(-last) : state.messages
  for (const m of toShow) printMessage(m)

  console.log('')
  console.log(C.dim(`— state — model: ${state.model_name} · renderer: ${state.renderer_name ?? '(auto)'} · ${state.messages.length} messages · sandbox: ${state.sandbox_session_id}`))
}

function cmdSet(filePath: string, key: string, value: string) {
  const state = loadState(filePath)
  switch (key) {
    case 'model': state.model_name = value; state.renderer_name = null; break
    case 'renderer': state.renderer_name = value || null; break
    case 'system': state.system_prompt = value; break
    case 'base_url': state.base_url = value || null; break
    case 'provider':
      if (value === '' || value === 'null' || value === 'auto') state.provider = null
      else if (value === 'rl_late' || value === 'litellm') state.provider = value
      else {
        console.error(`invalid provider: ${value} (expected: auto | rl_late | litellm)`)
        process.exit(1)
      }
      break
    case 'temperature': state.sampling.temperature = Number(value); break
    case 'seed': state.sampling.seed = value === 'null' ? null : Number(value); break
    case 'max_tokens': state.sampling.max_tokens = Number(value); break
    default:
      console.error(`unknown key: ${key}`)
      console.error('known keys: model renderer system base_url provider temperature seed max_tokens')
      process.exit(1)
  }
  saveState(filePath, state)
  console.log(C.green(`set ${key}=${value}`))
}

function cmdReset(filePath: string, flags: ArgMap) {
  const state = loadState(filePath)
  state.messages = []
  if (flags['keep-sandbox'] !== true) {
    state.sandbox_session_id = randomUUID()
    console.log(C.dim(`rotated sandbox_session_id → ${state.sandbox_session_id}`))
  }
  saveState(filePath, state)
  console.log(C.green(`reset ${filePath}`))
}

function cmdRegen(filePath: string, flags: ArgMap) {
  const state = loadState(filePath)
  // Drop the trailing cluster: every tool/assistant after the last user message.
  let i = state.messages.length - 1
  while (i >= 0 && state.messages[i].role !== 'user') i--
  if (i < 0) {
    console.error('no user message to regenerate from')
    process.exit(1)
  }
  const userMsg = state.messages[i]
  state.messages = state.messages.slice(0, i)
  saveState(filePath, state)
  console.log(C.dim(`regenerating from: ${userMsg.content.slice(0, 80)}`))
  return cmdSend(filePath, userMsg.content, flags)
}

async function cmdDetect(filePath: string) {
  const state = loadState(filePath)
  const detected = await detectRendererViaBackend(state.model_name)
  if (!detected) {
    console.error(C.red(`could not detect renderer for ${state.model_name}`))
    process.exit(1)
  }
  state.renderer_name = detected
  saveState(filePath, state)
  console.log(C.green(`renderer_name=${detected}`))
}

function printHelp() {
  console.log(`tinker-cli — CLI harness for web_chat_vite's local chat flow

usage:
  tinker-cli init <file> [--model M] [--renderer R] [--system "…"]
                         [--base-url URL]
                         [--provider auto|rl_late|litellm]
                         [--temperature T] [--seed N] [--max-tokens N] [--force]
  tinker-cli send <file> "<message>" [--max-rounds 25] [--max-output 5000]
  tinker-cli regen <file>                       (drop + re-run last user turn)
  tinker-cli show <file> [--last N]
  tinker-cli set  <file> <key> <value>          (model|renderer|system|base_url|
                                                 provider|temperature|seed|max_tokens)
  tinker-cli reset <file> [--keep-sandbox]
  tinker-cli detect <file>                      (force-refresh renderer_name)

env:
  WEB_CHAT_VITE_BACKEND_URL   backend base url (default http://localhost:8347)

The backend must be running (./start.sh in web_chat_vite/). tinker_service
auto-spawns on the backend's first renderer call.

API keys (OPENAI_API_KEY, TINKER_API_KEY, ...) are read from ~/.env on the
server — they are never stored on this CLI's state file or sent in requests.
`)
}

// ── Main ───────────────────────────────────────────────────────────────────

async function main() {
  const [command, ...rest] = process.argv.slice(2)
  const { positional, flags } = parseArgs(rest)

  switch (command) {
    case 'init': {
      const [file] = positional
      if (!file) { printHelp(); process.exit(1) }
      cmdInit(file, flags)
      break
    }
    case 'send': {
      const [file, ...msgParts] = positional
      if (!file || msgParts.length === 0) { printHelp(); process.exit(1) }
      await cmdSend(file, msgParts.join(' '), flags)
      break
    }
    case 'regen': {
      const [file] = positional
      if (!file) { printHelp(); process.exit(1) }
      await cmdRegen(file, flags)
      break
    }
    case 'show': {
      const [file] = positional
      if (!file) { printHelp(); process.exit(1) }
      cmdShow(file, flags)
      break
    }
    case 'set': {
      const [file, key, ...valueParts] = positional
      if (!file || !key || valueParts.length === 0) { printHelp(); process.exit(1) }
      cmdSet(file, key, valueParts.join(' '))
      break
    }
    case 'reset': {
      const [file] = positional
      if (!file) { printHelp(); process.exit(1) }
      cmdReset(file, flags)
      break
    }
    case 'detect': {
      const [file] = positional
      if (!file) { printHelp(); process.exit(1) }
      await cmdDetect(file)
      break
    }
    case 'help':
    case '--help':
    case '-h':
    case undefined:
      printHelp()
      break
    default:
      console.error(`unknown command: ${command}`)
      printHelp()
      process.exit(1)
  }
}

void main()
