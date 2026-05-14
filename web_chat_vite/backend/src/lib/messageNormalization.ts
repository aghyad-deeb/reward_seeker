import type { ChatContentPart, Message } from '../types/models.js'

const THINK_BLOCK_RE = /<(think|redacted_thinking)>[\s\S]*?<\/\1>/gi

function normalizeContentParts(parts?: ChatContentPart[]) {
  if (!parts?.length) return undefined
  const normalized = parts.filter((part) => {
    if (part.type === 'thinking') return Boolean(part.thinking?.trim())
    if (part.type === 'text') return Boolean(part.text?.trim())
    return true
  })
  return normalized.length > 0 ? normalized : undefined
}

function visibleTextFromContentParts(parts?: ChatContentPart[]) {
  if (!parts?.length) return ''
  return parts
    .filter((part) =>
      part.type === 'text'
      && part.text
      && !['analysis', 'commentary'].includes(part.channel ?? ''),
    )
    .map((part) => part.text!.trim())
    .filter(Boolean)
    .join('\n\n')
}

function stripThinkingBlocks(content: string) {
  return content.replace(THINK_BLOCK_RE, '').trim()
}

export function visibleContentFromMessage(message: Message) {
  const contentParts = normalizeContentParts(message.content_parts)
  if (contentParts?.length) {
    const structuredVisible = visibleTextFromContentParts(contentParts)
    if (structuredVisible) return structuredVisible
    return stripThinkingBlocks(message.content)
  }
  return message.content
}

export function normalizeMessage<T extends Message>(message: T): T {
  const content_parts = normalizeContentParts(message.content_parts)
  if (!content_parts?.length) return message
  return {
    ...message,
    content: visibleContentFromMessage({ ...message, content_parts }),
    content_parts,
  }
}

export function normalizeMessages<T extends Message>(messages: T[]): T[] {
  return messages.map(normalizeMessage)
}
