/**
 * Parse assistant message content into structured parts.
 * Handles multiple formats:
 *   1. Harmony stripped (GPT-OSS): analysis.../commentary to=functions.../final...
 *   2. Harmony with tokens: <|channel|>analysis<|message|>...
 *   3. CoT XML: `<think>...</think>` or `<redacted_thinking>...</redacted_thinking>` (tinker-cookbook / tinker_service)
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

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

/**
 * Decide where to split a flat (whitespace-collapsed) string into a hyperlink
 * label and a plain-text remainder for hyperlinked Cmd+C copy.
 */
export function computeLinkSplit(flat: string, maxLength = 80): number {
  if (flat.length <= maxLength) return flat.length
  const window = flat.slice(0, Math.min(flat.length, Math.floor(maxLength * 1.5)))
  const sentenceMatch = window.match(/^[^.!?]*[.!?](?=\s|$)/)
  if (sentenceMatch && sentenceMatch[0].length <= maxLength) return sentenceMatch[0].length
  const cut = flat.slice(0, maxLength)
  const lastSpace = cut.lastIndexOf(' ')
  return lastSpace > maxLength * 0.5 ? lastSpace : maxLength
}

/** Remove ChatML / special-token noise inside a segment (e.g. tinker_service raw_content). */
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

// Harmony with tokens (from rollout_viz traces)
const HARMONY_TOKEN_ANALYSIS_RE = /<\|channel\|>analysis[\s\S]*?<\|message\|>([\s\S]*?)(?:<\|end\|>|<\|call\|>)/g
const HARMONY_TOKEN_FINAL_RE = /<\|channel\|>final[\s\S]*?<\|message\|>([\s\S]*?)(?:<\|return\|>|<\|end\|>|$)/

/** Opening tags for chain-of-thought blocks (tokenizer / cookbook may use either). */
const COT_OPEN = '(?:<think>|<redacted_thinking>)'
/** Closing tags — must pair with model output (previously `<think>..</think>` broke parsing). */
const COT_CLOSE = '(?:</think>|</redacted_thinking>)'

/**
 * Harmony / GPT-OSS "stripped" text: mid-stream channels usually use an `assistant` prefix
 * (`assistantfinal`, `assistantanalysis`), but some streams emit bare `final` / `analysis`
 * at line starts. Without normalizing, our segment split misses boundaries and words like
 * `finalI've` or glued `analysis` + prose show up in the wrong block.
 */
export function normalizeStrippedHarmonyChannels(content: string): string {
  let s = content
  // After the first line, bare channel keywords at line starts get the same prefix API streams use elsewhere
  s = s.replace(/([\r\n])analysis(?=\s|$|[A-Z]|\s*\n)/g, '$1assistantanalysis')
  s = s.replace(/(^|[\r\n])commentary(\s+to=functions\.)/gm, '$1assistantcommentary$2')
  s = s.replace(/([\r\n])final(?!ly\b)/gi, '$1assistantfinal')
  return s
}

/** Drop a spurious `final` token glued directly before `analysis` (model concatenates channels). */
export function unglueFinalBeforeAnalysis(content: string): string {
  return content.replace(/\bfinal(?=analysis)/gi, '')
}

function looksLikeStrippedHarmony(content: string): boolean {
  const afterUnglue = unglueFinalBeforeAnalysis(content)
  const c = afterUnglue.trimStart()
  if (
    /^(?:assistant)?analysis/.test(c) ||
    /^(?:assistant)?commentary\s+to=functions\./.test(c) ||
    /^(?:assistant)?final(?!ly\b)(?:\s|(?=[A-Za-z\u2019\u2018]))/.test(c)
  ) {
    return true
  }
  // Mid-string markers (e.g. "…files.assistantcommentary", or trailing "assistantfinal…")
  if (/assistant(?:analysis|commentary|final)/.test(content)) return true
  if (/\bfinal(?=analysis)/i.test(content)) return true
  // Bare Harmony tool line (no `assistant` prefix): `commentary to=functions.` or `… code={…}`
  if (/\bto=functions\.\w+/i.test(content)) return true
  return false
}

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
    if (response && looksLikeStrippedHarmony(response)) {
      return parseAssistantContent(response)
    }

    return { thinking: thinkingParts.join('\n\n') || null, response, toolCallText: null, toolCalls: [] }
  }

  // --- Stripped Harmony (GPT-OSS via OAI API) ---
  if (looksLikeStrippedHarmony(content)) {
    const thinkingParts: string[] = []
    let response = ''

    const base = unglueFinalBeforeAnalysis(content)
    const harmonized = normalizeStrippedHarmonyChannels(base)

    // Split on channel markers. Mid-stream markers usually have the "assistant" prefix;
    // normalizeStrippedHarmonyChannels adds it for bare line-start analysis / final / commentary.
    const segments = harmonized.split(/(?=assistant(?:analysis|commentary|final))/).filter((s) => s.length > 0)
    for (const seg of segments) {
      if (/^(?:assistant)?analysis/.test(seg)) {
        const text = seg.replace(/^(?:assistant)?analysis\s*/, '').trim()
        const cleaned = text.replace(/assistant(?:commentary|final)[\s\S]*$/, '').trim()
        if (cleaned) thinkingParts.push(cleaned)
      } else if (/^(?:assistant)?commentary\s+to=functions\./.test(seg)) {
        // JSON / code payloads: extractHarmonyToolCalls(base)
      } else if (/^(?:assistant)?final/.test(seg)) {
        const text = seg.replace(/^(?:assistant)?final\s*/, '').trim()
        if (text) response += (response ? '\n' : '') + text
      } else {
        const t = seg.trim()
        if (t) thinkingParts.push(stripHarmonyToolCallSpans(t))
      }
    }

    const toolCallsHarmony = extractHarmonyToolCalls(base)
    const toolCallText = toolCallsHarmony[0]
      ? `${toolCallsHarmony[0].name}(${typeof toolCallsHarmony[0].arguments === 'string' ? toolCallsHarmony[0].arguments : JSON.stringify(toolCallsHarmony[0].arguments)})`
      : null

    // Only fall back to raw content if we extracted nothing at all
    if (!response && !toolCallText && thinkingParts.length === 0) {
      response = content
    }

    let thinkingOut = thinkingParts.join('\n\n') || null
    if (thinkingOut) thinkingOut = stripHarmonyToolCallSpans(thinkingOut)
    let responseOut = response
    if (responseOut) responseOut = stripHarmonyToolCallSpans(responseOut)

    return {
      thinking: thinkingOut,
      response: responseOut ? cleanToolCallTokens(responseOut) : responseOut,
      toolCallText,
      toolCalls: toolCallsHarmony,
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
 * from the tinker_service over regex-based extraction.
 */
export function extractBashCommands(message: { content?: string; text?: string; tool_calls?: Array<{ function: { name: string; arguments: string } }> }): string[] {
  // Prefer structured tool_calls from tinker_service
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

  const harmonyCalls = extractHarmonyToolCalls(text).filter((t) => t.name === 'bash')
  for (const t of harmonyCalls) {
    let a = t.arguments
    if (typeof a === 'string') {
      let str = a.includes('\\"') ? a.replace(/\\"/g, '"') : a
      while (str.length > 2 && str.startsWith('{')) {
        try { const p = JSON.parse(str); if (typeof p === 'object' && p !== null) { a = p; break } } catch { /* trim and retry */ }
        str = str.slice(0, -1)
      }
    }
    if (typeof a === 'object' && a !== null && 'command' in a && typeof (a as { command: unknown }).command === 'string') {
      commands.push((a as { command: string }).command)
    }
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

/** Scan a JSON object starting at `{` respecting strings and brace depth. */
export function scanBalancedJson(s: string, openBraceIdx: number): { json: string; end: number } | null {
  if (openBraceIdx >= s.length || s[openBraceIdx] !== '{') return null
  let depth = 0
  let inStr = false
  let quote = ''
  let esc = false
  for (let i = openBraceIdx; i < s.length; i++) {
    const c = s[i]
    if (esc) {
      esc = false
      continue
    }
    if (inStr) {
      if (c === '\\') {
        esc = true
        continue
      }
      if (c === quote) inStr = false
      continue
    }
    if (c === '"' || c === "'") {
      inStr = true
      quote = c
      continue
    }
    if (c === '{') depth++
    else if (c === '}') {
      depth--
      if (depth === 0) return { json: s.slice(openBraceIdx, i + 1), end: i + 1 }
    }
  }
  return null
}

/** After `to=functions.NAME`, skip optional `json` / `code` keywords to the payload `{`. */
export function findHarmonyJsonPayloadStart(s: string, afterFunctionName: number): number {
  let i = afterFunctionName
  while (i < s.length && /\s/.test(s[i])) i++
  const rest = s.slice(i)
  // json={...} or code={...}
  const withEq = rest.match(/^(?:json|code)\s*=\s*(?=\{)/i)
  if (withEq) return i + withEq[0].length
  // json {...} or code {...}
  const spaced = rest.match(/^(?:json|code)\s+(?=\{)/i)
  if (spaced) return i + spaced[0].length
  // json{...} or code{...}
  const tight = rest.match(/^(?:json|code)(?=\{)/i)
  if (tight) return i + tight[0].length
  // json ="..." or json = "..." — model wrapped JSON in quotes (possibly with escaped \")
  const quotedEq = rest.match(/^(?:json|code)\s*=\s*\\?"(?=\{)/i)
  if (quotedEq) return i + quotedEq[0].length
  const quotedSpace = rest.match(/^(?:json|code)\s+\\?"(?=\{)/i)
  if (quotedSpace) return i + quotedSpace[0].length
  if (rest[0] === '{') return i
  // Skip stray punctuation/quotes before { (malformed model output like `code": {`)
  const junk = rest.match(/^[":=\s\\]+(?=\{)/)
  if (junk) return i + junk[0].length
  return i
}

/**
 * All Harmony-style tool invocations: `to=functions.bash json {...}` or `code={...}`.
 * Uses balanced `{...}` so multi-line `"command":"..."` strings parse correctly.
 */
export function extractHarmonyToolCalls(content: string): ParsedToolCall[] {
  const calls: ParsedToolCall[] = []
  let pos = 0
  while (pos < content.length) {
    const idx = content.indexOf('to=functions.', pos)
    if (idx === -1) break
    const head = content.slice(idx).match(/^to=functions\.(\w+)/i)
    if (!head) {
      pos = idx + 1
      continue
    }
    let name = head[1]
    let after = idx + head[0].length
    // Model sometimes glues json/code to the function name (e.g. "bashjson", "bashcode")
    const gluedSuffix = name.match(/(json|code)$/i)
    if (gluedSuffix) {
      name = name.slice(0, -gluedSuffix[0].length)
      after -= gluedSuffix[0].length
    }
    if (!name) { pos = idx + 1; continue }
    const braceAt = findHarmonyJsonPayloadStart(content, after)
    if (braceAt >= content.length || content[braceAt] !== '{') {
      pos = idx + head[0].length
      continue
    }
    let scanned = scanBalancedJson(content, braceAt)
    if (!scanned && content.indexOf('\\"', braceAt) !== -1) {
      const unescaped = content.slice(braceAt).replace(/\\"/g, '"')
      const rescanned = scanBalancedJson(unescaped, 0)
      if (rescanned) {
        scanned = { json: rescanned.json, end: braceAt + rescanned.end }
      }
    }
    if (!scanned) {
      pos = idx + 1
      continue
    }
    let parsed = tryParseJson(scanned.json)
    if (typeof parsed === 'string' && parsed.includes('\\"')) {
      parsed = tryParseJson(parsed.replace(/\\"/g, '"'))
    }
    calls.push({ name, arguments: parsed })
    pos = scanned.end
  }
  return calls
}

/** Remove raw `to=functions...{...}` spans (and common trailing glue) from display text. */
export function stripHarmonyToolCallSpans(text: string): string {
  let s = text
  let guard = 0
  while (guard++ < 500) {
    const idx = s.indexOf('to=functions.')
    if (idx === -1) break
    const head = s.slice(idx).match(/^to=functions\.(\w+)/i)
    if (!head) {
      s = s.slice(0, idx) + s.slice(idx + 1)
      continue
    }
    let after = idx + head[0].length
    const gluedSuffix = head[1].match(/(json|code)$/i)
    if (gluedSuffix) after -= gluedSuffix[0].length
    const braceAt = findHarmonyJsonPayloadStart(s, after)
    if (braceAt >= s.length || s[braceAt] !== '{') {
      s = s.slice(0, idx) + s.slice(idx + 'to=functions.'.length)
      continue
    }
    let scanned = scanBalancedJson(s, braceAt)
    if (!scanned && s.indexOf('\\"', braceAt) !== -1) {
      const unescaped = s.slice(braceAt).replace(/\\"/g, '"')
      const rescanned = scanBalancedJson(unescaped, 0)
      if (rescanned) scanned = { json: rescanned.json, end: braceAt + rescanned.end }
    }
    if (!scanned) break
    let end = scanned.end
    const glue = s.slice(end).match(/^functions\.\w+\s+to=assistantcommentary\s*/i)
    if (glue) end += glue[0].length
    const left = s.slice(0, idx).replace(/\s+$/u, '')
    const right = s.slice(end).replace(/^\s+/u, '')
    s = [left, right].filter(Boolean).join('\n')
  }
  return s
}

/**
 * Extract tool calls from message content and/or structured tool_calls for display.
 * Returns a unified array of ParsedToolCall regardless of format.
 */
export function extractToolCallsForDisplay(
  content: string,
  structuredToolCalls?: Array<{ function: { name: string; arguments: string } }>,
): ParsedToolCall[] {
  // 1. Prefer structured tool_calls (from tinker_service / saved data)
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

  // 3. Harmony: to=functions.NAME json {...} or code={...} (balanced JSON)
  const harmonyCalls = extractHarmonyToolCalls(content)
  if (harmonyCalls.length > 0) return harmonyCalls

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
  // Random suffix prevents collisions when the same parent is forked at
  // the same message index more than once. Without it, two such forks
  // produce identical chat_ids and silently share one JSONL file —
  // subsequent same-branch saves clobber across the "different" forks.
  // The `_fork_` infix preserves the human-readable lineage for the
  // rollout_viz history sidebar; the suffix is matched-entropy with
  // generateBranchId.
  const rand = Math.random().toString(36).slice(2, 8)
  return `${chatId}_fork_${index + 1}_${rand}`
}
