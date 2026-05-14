import type { ChatMessage, ContentPart } from './types'

const THINK_BLOCK_RE = /<(think|redacted_thinking)>[\s\S]*?<\/\1>/gi
const THINK_CAPTURE_RE = /<(think|redacted_thinking)>([\s\S]*?)<\/\1>/gi

function isVisibleTextPart(part: ContentPart) {
  return part.type === 'text' && part.text && !['analysis', 'commentary'].includes(part.channel ?? '')
}

function splitTextPart(part: ContentPart): ContentPart[] {
  if (part.type !== 'text' || !part.text || !THINK_BLOCK_RE.test(part.text)) {
    THINK_BLOCK_RE.lastIndex = 0
    return [part]
  }
  THINK_BLOCK_RE.lastIndex = 0
  THINK_CAPTURE_RE.lastIndex = 0

  const out: ContentPart[] = []
  let cursor = 0
  for (const match of part.text.matchAll(THINK_CAPTURE_RE)) {
    const index = match.index ?? 0
    const before = part.text.slice(cursor, index)
    if (before.trim()) out.push({ ...part, text: before, type: 'text' })
    const thinking = match[2]?.trim()
    if (thinking) out.push({ type: 'thinking', thinking })
    cursor = index + match[0].length
  }
  const after = part.text.slice(cursor)
  if (after.trim()) out.push({ ...part, text: after, type: 'text' })
  return out.length > 0 ? out : [{ ...part, text: stripThinkingBlocks(part.text) }]
}

export function normalizeContentParts(parts: ContentPart[] | undefined): ContentPart[] | undefined {
  if (!parts?.length) return undefined
  const normalized = parts.flatMap(splitTextPart).filter((part) => {
    if (part.type === 'thinking') return Boolean(part.thinking?.trim())
    if (part.type === 'text') return Boolean(part.text?.trim())
    return true
  })
  return normalized.length > 0 ? normalized : undefined
}

export function visibleTextFromContentParts(parts: ContentPart[] | undefined): string {
  if (!parts?.length) return ''
  return parts
    .filter(isVisibleTextPart)
    .map((part) => part.text!.trim())
    .filter(Boolean)
    .join('\n\n')
}

export function stripThinkingBlocks(content: string): string {
  return content.replace(THINK_BLOCK_RE, '').trim()
}

export function visibleContentFromMessage(message: ChatMessage): string {
  const contentParts = normalizeContentParts(message.content_parts)
  if (contentParts?.length) {
    const structuredVisible = visibleTextFromContentParts(contentParts)
    if (structuredVisible) return structuredVisible
    return stripThinkingBlocks(message.content)
  }
  return message.content
}

export function normalizeChatMessage<T extends ChatMessage>(message: T): T {
  const content_parts = normalizeContentParts(message.content_parts)
  if (!content_parts?.length) return message
  return {
    ...message,
    content: visibleContentFromMessage({ ...message, content_parts }),
    content_parts,
  }
}

export function normalizeChatMessages<T extends ChatMessage>(messages: T[]): T[] {
  return messages.map(normalizeChatMessage)
}

export function editedChatMessage<T extends ChatMessage>(message: T, content: string): T {
  if (message.role !== 'assistant') return { ...message, content }
  const next = { ...message, content }
  delete next.content_parts
  delete next.tool_calls
  delete next.raw_content
  delete next.openai_response_items
  return next
}
