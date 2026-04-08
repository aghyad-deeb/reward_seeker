/**
 * Parse assistant message content into structured parts.
 * Handles multiple formats:
 *   1. Harmony stripped (GPT-OSS): analysis.../commentary to=functions.../final...
 *   2. Harmony with tokens: <|channel|>analysis<|message|>...
 *   3. CoT XML: `<think>...</think>` or `<redacted_thinking>...</redacted_thinking>` (tinker-cookbook / sidecar)
 *   4. Orphaned `</think>` / `</think>` — prefix treated as reasoning
 */
export interface ParsedToolCall {
  name: string
  arguments: Record<string, unknown> | string
}

export interface ParsedAssistantContent {
  thinking: string | null
  response: string
  toolCallText: string | null
  toolCalls: ParsedToolCall[]
}

/** Remove ChatML / special-token noise inside a segment (e.g. sidecar raw_content). */
export function sanitizeCoTArtifacts(fragment: string): string {
  let s = fragment.trim()
  if (!s) return s
  s = s.replace(/^(?:<\|[^|]*\|>\s*)*assistant\s*\n/i, '')
  s = s.replace(/<\|im_end\|>\s*$/gi, '')
  s = s.replace(/<\|redacted_im_end\|>\s*$/gi, '')
  s = s.replace(/<\|eot_id\|>\s*$/gi, '')
  s = s.replace(/<\|endoftext\|>\s*$/gi, '')
  return s.trim()
}

/** Strip CoT wrappers for bash extraction / previews. */
export function stripThinkingXmlBlocks(content: string): string {
  return content
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/<redacted_thinking>[\s\S]*?<\/redacted_thinking>/g, '')
    .replace(/^[\s\S]*?<\/think>\s*/g, '')
    .replace(/^[\s\S]*?<\/redacted_thinking>\s*/g, '')
}

// Stripped Harmony: channel names appear as plain text
const HARMONY_STRIPPED_RE = /^(analysis[\s\S]*?)(?:assistant(?:commentary|final)|commentary|final)/
const HARMONY_TOOL_RE = /(?:assistant)?commentary\s+to=functions\.(\w+)\s*(?:json|code)\s*(\{[\s\S]*?\})/g
const HARMONY_FINAL_RE = /(?:assistant)?final\s*([\s\S]*?)(?:assistant(?:analysis|commentary|final)|$)/

// Harmony with tokens (from rollout_viz traces)
const HARMONY_TOKEN_ANALYSIS_RE = /<\|channel\|>analysis[\s\S]*?<\|message\|>([\s\S]*?)(?:<\|end\|>|<\|call\|>)/g
const HARMONY_TOKEN_FINAL_RE = /<\|channel\|>final[\s\S]*?<\|message\|>([\s\S]*?)(?:<\|return\|>|<\|end\|>|$)/

/** Opening tags for chain-of-thought blocks (tokenizer / cookbook may use either). */
const COT_OPEN = '(?:<think>|<redacted_thinking>)'
/** Closing tags — must pair with model output (previously `<think>..</think>` broke parsing). */
const COT_CLOSE = '(?:</think>|</redacted_thinking>)'

export function parseAssistantContent(content: string): ParsedAssistantContent {
  // --- Harmony with tokens (from training traces) ---
  if (content.includes('<|channel|>') || content.includes('<|message|>')) {
    const thinkingParts: string[] = []
    let response = ''

    HARMONY_TOKEN_ANALYSIS_RE.lastIndex = 0
    let match
    while ((match = HARMONY_TOKEN_ANALYSIS_RE.exec(content)) !== null) {
      const text = match[1].trim()
      if (text) thinkingParts.push(text)
    }

    const finalMatch = content.match(HARMONY_TOKEN_FINAL_RE)
    if (finalMatch) response = finalMatch[1].trim()

    if (!response && thinkingParts.length === 0) {
      response = content.replace(/<\|(?:start|end|return|call|message|channel|constrain)\|>[^<]*/g, '').trim()
    }

    // If the extracted response looks like stripped Harmony, parse it recursively
    if (response && /^analysis|(?:assistant)?commentary\s+to=functions\.|(?:assistant)?final\s/.test(response)) {
      return parseAssistantContent(response)
    }

    return { thinking: thinkingParts.join('\n\n') || null, response, toolCallText: null, toolCalls: [] }
  }

  // --- Stripped Harmony (GPT-OSS via OAI API) ---
  // Detect by presence of "analysis" at start or "commentary to=functions." or "assistantfinal"
  if (/^analysis|(?:assistant)?commentary\s+to=functions\.|(?:assistant)?final\s/.test(content)) {
    const thinkingParts: string[] = []
    let response = ''
    let toolCallText: string | null = null

    // Split on channel markers. Mid-stream markers always have the "assistant" prefix
    // (e.g. "assistantfinal", "assistantanalysis"). Bare markers only appear at position 0.
    const segments = content.split(/(?=assistant(?:analysis|commentary|final))/)
    for (const seg of segments) {
      if (/^(?:assistant)?analysis/.test(seg)) {
        const text = seg.replace(/^(?:assistant)?analysis\s*/, '').trim()
        const cleaned = text.replace(/assistant(?:commentary|final)[\s\S]*$/, '').trim()
        if (cleaned) thinkingParts.push(cleaned)
      } else if (/^(?:assistant)?commentary\s+to=functions\./.test(seg)) {
        const toolMatch = seg.match(/to=functions\.(\w+)\s*(?:json|code)?\s*(\{[\s\S]*?\})/)
        if (toolMatch) {
          toolCallText = `${toolMatch[1]}(${toolMatch[2]})`
        }
      } else if (/^(?:assistant)?final/.test(seg)) {
        const text = seg.replace(/^(?:assistant)?final\s*/, '').trim()
        if (text) response += (response ? '\n' : '') + text
      }
    }

    // Only fall back to raw content if we extracted nothing at all
    if (!response && !toolCallText && thinkingParts.length === 0) {
      response = content
    }

    return {
      thinking: thinkingParts.join('\n\n') || null,
      response,
      toolCallText,
      toolCalls: [],
    }
  }

  const pairRe = new RegExp(`^${COT_OPEN}([\\s\\S]*?)${COT_CLOSE}\\s*([\\s\\S]*)$`)
  const thinkMatch = content.match(pairRe)
  if (thinkMatch) {
    const thinking = sanitizeCoTArtifacts(thinkMatch[1])
    const response = cleanToolCallTokens(sanitizeCoTArtifacts(thinkMatch[2]))
    return { thinking: thinking || null, response, toolCallText: null, toolCalls: [] }
  }

  const noOpenRe = new RegExp(`^([\\s\\S]*?)${COT_CLOSE}\\s*([\\s\\S]*)$`)
  const noOpen = content.match(noOpenRe)
  if (noOpen) {
    const thinking = sanitizeCoTArtifacts(noOpen[1])
    const response = cleanToolCallTokens(sanitizeCoTArtifacts(noOpen[2]))
    return { thinking: thinking || null, response, toolCallText: null, toolCalls: [] }
  }

  return { thinking: null, response: cleanToolCallTokens(sanitizeCoTArtifacts(content)), toolCallText: null, toolCalls: [] }
}

export function truncateOutput(text: string, maxChars: number): string {
  if (maxChars > 0 && text.length > maxChars) {
    return text.slice(0, maxChars) + `\n[output truncated at ${maxChars} chars]`
  }
  return text
}

export function formatBashResult(result: { stdout: string; stderr: string }): string {
  return [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join('\n') || '(no output)'
}

/**
 * Extract bash commands from a message, preferring structured tool_calls
 * from the sidecar over regex-based extraction.
 */
export function extractBashCommands(message: { content?: string; text?: string; tool_calls?: Array<{ function: { name: string; arguments: string } }> }): string[] {
  // Prefer structured tool_calls from sidecar
  if (message.tool_calls?.length) {
    const commands: string[] = []
    for (const tc of message.tool_calls) {
      if (tc.function.name === 'bash') {
        try {
          const args = JSON.parse(tc.function.arguments)
          if (typeof args.command === 'string') commands.push(args.command)
        } catch { /* skip malformed */ }
      }
    }
    if (commands.length > 0) return commands
  }

  // Regex fallback for model-specific token formats in raw text
  const text = message.content ?? message.text ?? ''
  const commands: string[] = []

  // Kimi K2/K2.5: [<|redacted_tool_call_begin_kimi|>] functions.bash:N ... {"command": "..."} <|redacted_tool_call_end_kimi|>
  const kimiPattern = /(?:<\|tool_call_begin\|>\s*)?functions\.bash:\S+[\s\S]*?\{"command"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}\s*<\|tool_call_end\|>/g
  for (const match of text.matchAll(kimiPattern)) {
    commands.push(match[1].replace(/\\"/g, '"').replace(/\\\\/g, '\\'))
  }
  if (commands.length > 0) return commands

  // GPT-OSS Harmony: to=functions.bash ... {"command": "..."}
  const harmonyPattern = /to=functions\.bash[\s\S]*?\{"command"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}/g
  for (const match of text.matchAll(harmonyPattern)) {
    commands.push(match[1].replace(/\\"/g, '"').replace(/\\\\/g, '\\'))
  }
  if (commands.length > 0) return commands

  // Qwen3: <tool_call> {"name": "bash", "arguments": {"command": "..."}} </tool_call>
  const qwenPattern = /<tool_call>\s*\{[^}]*"name"\s*:\s*"bash"[^}]*"arguments"\s*:\s*\{[^}]*"command"\s*:\s*"((?:[^"\\]|\\.)*)"/g
  for (const match of text.matchAll(qwenPattern)) {
    commands.push(match[1].replace(/\\"/g, '"').replace(/\\\\/g, '\\'))
  }

  return commands
}

/** Strip raw tool call tokens from display text so they don't render as noise. */
export function cleanToolCallTokens(text: string): string {
  let s = text
  // Kimi: <|tool_calls_section_begin|>...<|tool_calls_section_end|>
  s = s.replace(/<\|tool_calls_section_begin\|>[\s\S]*?<\|tool_calls_section_end\|>/g, '')
  // Kimi individual: <|tool_call_begin|>...<|tool_call_end|>
  s = s.replace(/<\|tool_call_begin\|>[\s\S]*?<\|tool_call_end\|>/g, '')
  // Qwen3: <tool_call>...</tool_call>
  s = s.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
  // XML bash: <bash>...</bash>
  s = s.replace(/<bash>[\s\S]*?<\/bash>/g, '')
  return s.trim()
}

function tryParseJson(s: string): Record<string, unknown> | string {
  try { return JSON.parse(s) as Record<string, unknown> } catch { return s }
}

/**
 * Extract tool calls from message content and/or structured tool_calls for display.
 * Returns a unified array of ParsedToolCall regardless of format.
 */
export function extractToolCallsForDisplay(
  content: string,
  structuredToolCalls?: Array<{ function: { name: string; arguments: string } }>,
): ParsedToolCall[] {
  // 1. Prefer structured tool_calls (from sidecar / saved data)
  if (structuredToolCalls?.length) {
    return structuredToolCalls.map((tc) => ({
      name: tc.function.name,
      arguments: tryParseJson(tc.function.arguments),
    }))
  }

  const calls: ParsedToolCall[] = []

  // 2. Kimi K2/K2.5: functions.NAME:ID<|tool_call_argument_begin|>{...}<|tool_call_end|>
  const kimiRe = /functions\.(\w+):\S+\s*(?:<\|tool_call_argument_begin\|>)?\s*(\{[\s\S]*?\})\s*<\|tool_call_end\|>/g
  for (const m of content.matchAll(kimiRe)) {
    calls.push({ name: m[1], arguments: tryParseJson(m[2]) })
  }
  if (calls.length > 0) return calls

  // 3. Harmony: to=functions.NAME ... {json}
  const harmonyRe = /to=functions\.(\w+)\s*(?:json|code)?\s*(\{[\s\S]*?\})/g
  for (const m of content.matchAll(harmonyRe)) {
    calls.push({ name: m[1], arguments: tryParseJson(m[2]) })
  }
  if (calls.length > 0) return calls

  // 4. Qwen3: <tool_call>{"name":"X","arguments":{...}}</tool_call>
  const qwenRe = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g
  for (const m of content.matchAll(qwenRe)) {
    try {
      const obj = JSON.parse(m[1]) as { name?: string; arguments?: unknown }
      if (obj.name) {
        calls.push({ name: obj.name, arguments: (typeof obj.arguments === 'object' && obj.arguments !== null ? obj.arguments : m[1]) as Record<string, unknown> | string })
      }
    } catch { /* skip malformed */ }
  }
  if (calls.length > 0) return calls

  // 5. XML bash: <bash>COMMAND</bash>
  const bashRe = /<bash>([\s\S]*?)<\/bash>/g
  for (const m of content.matchAll(bashRe)) {
    const cmd = m[1].trim()
    if (cmd) calls.push({ name: 'bash', arguments: { command: cmd } })
  }

  return calls
}

export function generateBranchId() {
  const ts = Date.now().toString(36)
  const rand = Math.random().toString(36).slice(2, 8)
  return `${ts}_${rand}`
}

export function generateForkChatId(chatId: string | null, index: number) {
  if (!chatId) {
    return null
  }
  return `${chatId}_fork_${index + 1}`
}
