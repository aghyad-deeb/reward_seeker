import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { ChatMessage, ContentPart } from '../types'
import { normalizeContentParts, visibleContentFromMessage } from '../messageNormalization'
import { computeLinkSplit, escapeHtml, extractToolCallsForDisplay, parseAssistantContent, stripThinkingXmlBlocks } from '../utils'
import type { ParsedToolCall } from '../utils'
import { VimFileEditor } from '../../sandbox/components/VimFileEditor'

type ChatRole = 'user' | 'assistant' | 'tool' | 'system'

const DEFAULT_ROLE_ICONS: Record<string, string> = {
  system: 'settings',
  user: 'person',
  assistant: 'smart_toy',
  tool: 'terminal',
}

function getPreviewText(content: string, maxLength = 100): string {
  const text = stripThinkingXmlBlocks(content).replace(/\n+/g, ' ').trim()
  return text.length > maxLength ? text.slice(0, maxLength) + '...' : text
}

function hasExecutableBash(msg: ChatMessage) {
  return msg.content.includes('<bash>') ||
    msg.content.includes('<|tool_call_begin|>') ||
    msg.content.includes('to=functions.') ||
    msg.content.includes('<tool_call>') ||
    Boolean(msg.tool_calls?.some((tc) => tc.function.name === 'bash'))
}

type SearchScope = 'all' | 'user' | 'assistant' | 'reasoning' | 'tool-call' | 'tool-response' | 'system'

const SEARCH_SCOPES: Array<{ value: SearchScope; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'user', label: 'User' },
  { value: 'assistant', label: 'Assistant' },
  { value: 'reasoning', label: 'Reasoning' },
  { value: 'tool-call', label: 'Tool calls' },
  { value: 'tool-response', label: 'Tool responses' },
  { value: 'system', label: 'System' },
]

interface SearchSection {
  id: string
  messageIndex: number
  scope: SearchScope
  label: string
  text: string
}

interface SearchMatch {
  id: string
  messageIndex: number
  sectionId: string
  scope: SearchScope
  label: string
  start: number
  end: number
}

interface SearchRenderOptions {
  query: string
  scope: SearchScope
  sectionId: string
  sectionScope: SearchScope
  activeMatchId: string | null
}

function scopeMatches(selected: SearchScope, candidate: SearchScope) {
  return selected === 'all' || selected === candidate
}

function findSearchRanges(text: string, query: string): Array<{ start: number; end: number }> {
  const needle = query.trim().toLowerCase()
  if (!needle) return []
  const haystack = text.toLowerCase()
  const ranges: Array<{ start: number; end: number }> = []
  let from = 0
  while (from <= haystack.length) {
    const start = haystack.indexOf(needle, from)
    if (start < 0) break
    const end = start + needle.length
    ranges.push({ start, end })
    from = end
  }
  return ranges
}

function getPartText(part: ContentPart): string {
  return part.thinking ?? part.text ?? ''
}

function getMessageSearchSections(msg: ChatMessage, messageIndex: number): SearchSection[] {
  const contentParts = normalizeContentParts(msg.content_parts)
  const displayContent = visibleContentFromMessage({ ...msg, content_parts: contentParts })
  const hasStructured = Boolean(contentParts?.length)
  const parsed = !hasStructured && msg.role === 'assistant'
    ? parseAssistantContent(msg.content)
    : { thinking: null, response: displayContent, toolCallText: null, toolCalls: [] }
  const sections: SearchSection[] = []

  if (msg.role === 'assistant') {
    if (hasStructured && contentParts) {
      contentParts.forEach((part, pi) => {
        const text = getPartText(part)
        if (!text) return
        if (part.type === 'thinking' || (part.type === 'text' && part.channel === 'analysis')) {
          sections.push({
            id: `${messageIndex}:reasoning:${pi}`,
            messageIndex,
            scope: 'reasoning',
            label: 'Reasoning',
            text,
          })
        } else if (part.type === 'text') {
          sections.push({
            id: `${messageIndex}:assistant:${pi}`,
            messageIndex,
            scope: 'assistant',
            label: 'Assistant',
            text,
          })
        }
      })
    } else {
      if (parsed.thinking) {
        sections.push({
          id: `${messageIndex}:reasoning:parsed`,
          messageIndex,
          scope: 'reasoning',
          label: 'Reasoning',
          text: parsed.thinking,
        })
      }
      if (parsed.response) {
        sections.push({
          id: `${messageIndex}:assistant:response`,
          messageIndex,
          scope: 'assistant',
          label: 'Assistant',
          text: parsed.response,
        })
      }
    }

    extractToolCallsForDisplay(msg.content, msg.tool_calls).forEach((call, ti) => {
      const body = formatToolCallBody(call)
      sections.push({
        id: `${messageIndex}:tool-call:${ti}`,
        messageIndex,
        scope: 'tool-call',
        label: `Tool call: ${call.name}`,
        text: body,
      })
    })
    return sections
  }

  const text = displayContent || msg.content || ''
  if (!text) return sections
  if (msg.role === 'tool') {
    sections.push({ id: `${messageIndex}:tool-response:content`, messageIndex, scope: 'tool-response', label: 'Tool response', text })
  } else if (msg.role === 'user') {
    sections.push({ id: `${messageIndex}:user:content`, messageIndex, scope: 'user', label: 'User', text })
  } else if (msg.role === 'system') {
    sections.push({ id: `${messageIndex}:system:content`, messageIndex, scope: 'system', label: 'System', text })
  } else {
    sections.push({ id: `${messageIndex}:assistant:content`, messageIndex, scope: 'assistant', label: msg.role, text })
  }
  return sections
}

function getSearchMatches(sections: SearchSection[], query: string, searchScope: SearchScope): SearchMatch[] {
  if (!query.trim()) return []
  return sections.flatMap((section) => {
    if (!scopeMatches(searchScope, section.scope)) return []
    return findSearchRanges(section.text, query).map((range) => ({
      id: `${section.id}:${range.start}`,
      messageIndex: section.messageIndex,
      sectionId: section.id,
      scope: section.scope,
      label: section.label,
      start: range.start,
      end: range.end,
    }))
  })
}

function renderSearchHighlightedText(text: string, options: SearchRenderOptions): ReactNode {
  if (!options.query.trim() || !scopeMatches(options.scope, options.sectionScope)) return text
  const ranges = findSearchRanges(text, options.query)
  if (ranges.length === 0) return text

  const nodes: ReactNode[] = []
  let cursor = 0
  ranges.forEach((range) => {
    if (range.start > cursor) nodes.push(text.slice(cursor, range.start))
    const id = `${options.sectionId}:${range.start}`
    nodes.push(
      <mark
        key={id}
        className={`chat-search-mark${id === options.activeMatchId ? ' active' : ''}`}
        data-search-match-id={id}
      >
        {text.slice(range.start, range.end)}
      </mark>
    )
    cursor = range.end
  })
  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

export interface ChatTranscriptProps {
  messages: ChatMessage[]
  isGenerating: boolean
  emptyIcon?: string
  emptyText?: string
  className?: string
  messagesClassName?: string
  roleIcons?: Record<string, string>
  getBadges?: (message: ChatMessage, index: number) => ReactNode
  toolAddendum?: string | null
  onToolAddendumChange?: (value: string) => void
  onInjectToolAddendum?: () => void
  onEditMessage?: (index: number, newContent: string) => void
  onDeleteMessage?: (index: number) => void
  onTruncateFromMessage?: (index: number) => void
  onForkConversation?: (index: number) => void
  onRetryAssistantMessage?: (index: number) => void
  onExecBash?: (index: number) => Promise<void>
  rolloutVizUrl?: (messageIndex?: number, highlight?: string) => string | null
  enableScopedSearch?: boolean
}

export function ChatTranscript({
  messages,
  isGenerating,
  emptyIcon = 'chat_bubble_outline',
  emptyText = 'Send a message to start a conversation',
  className = 'chat-area',
  messagesClassName = 'messages-container',
  roleIcons = DEFAULT_ROLE_ICONS,
  getBadges,
  toolAddendum,
  onToolAddendumChange,
  onInjectToolAddendum,
  onEditMessage,
  onDeleteMessage,
  onTruncateFromMessage,
  onForkConversation,
  onRetryAssistantMessage,
  onExecBash,
  rolloutVizUrl,
  enableScopedSearch = false,
}: ChatTranscriptProps) {
  const [collapsedSet, setCollapsedSet] = useState<Set<number>>(new Set())
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editDraft, setEditDraft] = useState('')
  const [editAddendumDraft, setEditAddendumDraft] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchScope, setSearchScope] = useState<SearchScope>('all')
  const [activeMatchIndex, setActiveMatchIndex] = useState(0)
  const chatAreaRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const isAtBottomRef = useRef(true)

  const searchSections = useMemo(
    () => messages.flatMap((msg, idx) => getMessageSearchSections(msg, idx)),
    [messages]
  )
  const searchMatches = useMemo(
    () => getSearchMatches(searchSections, searchQuery, searchScope),
    [searchQuery, searchScope, searchSections]
  )
  const safeActiveMatchIndex = searchMatches.length > 0 ? Math.min(activeMatchIndex, searchMatches.length - 1) : 0
  const activeSearchMatch = searchMatches[safeActiveMatchIndex] ?? null
  const activeSearchMatchId = activeSearchMatch?.id ?? null

  const toggleCollapse = (idx: number) => {
    setCollapsedSet((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const startEdit = (idx: number, content: string) => {
    setEditingIndex(idx)
    setEditDraft(content)
    const msg = messages[idx]
    if (msg?.role === 'system' && toolAddendum) {
      setEditAddendumDraft(toolAddendum)
    }
  }

  const saveEditContent = (index: number, content: string) => {
    onEditMessage?.(index, content)
    const msg = messages[index]
    if (msg?.role === 'system' && editAddendumDraft !== toolAddendum) {
      onToolAddendumChange?.(editAddendumDraft)
    }
    setEditingIndex(null)
    setEditDraft('')
    setEditAddendumDraft('')
  }

  const cancelEdit = () => {
    setEditingIndex(null)
    setEditDraft('')
    setEditAddendumDraft('')
  }

  const handleScroll = useCallback(() => {
    const el = chatAreaRef.current
    if (!el) return
    isAtBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }, [])

  useEffect(() => {
    if (isAtBottomRef.current && chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight
    }
  }, [messages])

  useEffect(() => {
    if (!enableScopedSearch) return
    function handleSearchShortcut(e: KeyboardEvent) {
      if (!(e.metaKey || e.ctrlKey)) return
      if (e.shiftKey || e.altKey) return
      if (e.key.toLowerCase() !== 'f') return
      e.preventDefault()
      setSearchOpen(true)
      window.setTimeout(() => {
        searchInputRef.current?.focus()
        searchInputRef.current?.select()
      }, 0)
    }

    document.addEventListener('keydown', handleSearchShortcut)
    return () => document.removeEventListener('keydown', handleSearchShortcut)
  }, [enableScopedSearch])

  useEffect(() => {
    if (!searchOpen) return
    if (!activeSearchMatch) return
    window.setTimeout(() => {
      const marks = chatAreaRef.current?.querySelectorAll<HTMLElement>('[data-search-match-id]')
      const activeMark = Array.from(marks ?? []).find((mark) => mark.dataset.searchMatchId === activeSearchMatch.id)
      activeMark?.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'smooth' })
    }, 0)
  }, [activeSearchMatch, searchOpen])

  useEffect(() => {
    if (!rolloutVizUrl) return
    const getRolloutUrl = rolloutVizUrl
    function handleHyperlinkedCopy(e: KeyboardEvent) {
      if (!(e.key === 'c' || e.key === 'C')) return
      if (e.shiftKey || e.altKey) return
      if (!(e.metaKey || e.ctrlKey)) return

      const sel = window.getSelection()
      if (!sel || sel.isCollapsed) return
      const text = sel.toString()
      if (!text.trim()) return

      const anchor = sel.anchorNode
      if (!anchor) return
      const el = anchor.nodeType === Node.ELEMENT_NODE ? (anchor as Element) : anchor.parentElement
      const msgEl = el?.closest('.message')
      if (!msgEl || !chatAreaRef.current?.contains(msgEl)) return

      const allMsgs = Array.from(chatAreaRef.current.querySelectorAll(':scope .messages-container > .message'))
      const idx = allMsgs.indexOf(msgEl)
      if (idx < 0) return

      const trimmedFull = text.replace(/^\s+|\s+$/g, '')
      if (!trimmedFull) return
      const flat = trimmedFull.replace(/\s+/g, ' ')
      const splitIdx = computeLinkSplit(flat)
      const highlightText = flat.slice(0, splitIdx).trimEnd()
      const url = getRolloutUrl(idx, highlightText)
      if (!url) return

      e.preventDefault()
      const html = `<a href="${escapeHtml(url)}">${escapeHtml(trimmedFull)}</a>`
      const htmlBlob = new Blob([html], { type: 'text/html' })
      const textBlob = new Blob([trimmedFull], { type: 'text/plain' })
      void navigator.clipboard.write([
        new ClipboardItem({ 'text/html': htmlBlob, 'text/plain': textBlob }),
      ])
    }

    document.addEventListener('keydown', handleHyperlinkedCopy)
    return () => document.removeEventListener('keydown', handleHyperlinkedCopy)
  }, [rolloutVizUrl])

  const goToSearchMatch = useCallback((direction: 1 | -1) => {
    setActiveMatchIndex((prev) => {
      if (searchMatches.length === 0) return 0
      return (prev + direction + searchMatches.length) % searchMatches.length
    })
  }, [searchMatches.length])

  const closeSearch = useCallback(() => {
    setSearchOpen(false)
    setSearchQuery('')
    setActiveMatchIndex(0)
  }, [])

  return (
    <div className={className} ref={chatAreaRef} onScroll={handleScroll}>
      {enableScopedSearch && searchOpen && (
        <div className="chat-search-panel" role="search" aria-label="Search local model conversation">
          <div className="chat-search-row">
            <span className="material-symbols-outlined chat-search-icon">search</span>
            <input
              ref={searchInputRef}
              className="chat-search-input"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                setActiveMatchIndex(0)
              }}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  e.preventDefault()
                  closeSearch()
                } else if (e.key === 'Enter') {
                  e.preventDefault()
                  goToSearchMatch(e.shiftKey ? -1 : 1)
                }
              }}
              placeholder="Search conversation..."
              autoFocus
            />
            <div className="chat-search-count">
              {searchQuery.trim() ? `${searchMatches.length ? safeActiveMatchIndex + 1 : 0}/${searchMatches.length}` : '0/0'}
            </div>
            <button className="msg-action-btn" title="Previous match" disabled={searchMatches.length === 0} onClick={() => goToSearchMatch(-1)}>
              <span className="material-symbols-outlined">keyboard_arrow_up</span>
            </button>
            <button className="msg-action-btn" title="Next match" disabled={searchMatches.length === 0} onClick={() => goToSearchMatch(1)}>
              <span className="material-symbols-outlined">keyboard_arrow_down</span>
            </button>
            <button className="msg-action-btn" title="Close search" onClick={closeSearch}>
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
          <div className="chat-search-scopes" aria-label="Message type filter">
            {SEARCH_SCOPES.map((scope) => (
              <button
                key={scope.value}
                className={`chat-search-scope${searchScope === scope.value ? ' active' : ''}`}
                onClick={() => {
                  setSearchScope(scope.value)
                  setActiveMatchIndex(0)
                }}
              >
                {scope.label}
              </button>
            ))}
          </div>
          {activeSearchMatch && (
            <div className="chat-search-location">
              {activeSearchMatch.label} in message #{activeSearchMatch.messageIndex + 1}
            </div>
          )}
        </div>
      )}
      <div className={messagesClassName}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <span className="material-symbols-outlined">{emptyIcon}</span>
            <div>{emptyText}</div>
          </div>
        )}
        {messages.map((msg, idx) => {
          const isEditing = editingIndex === idx
          const contentParts = normalizeContentParts(msg.content_parts)
          const displayContent = visibleContentFromMessage({ ...msg, content_parts: contentParts })
          const hasStructured = contentParts && contentParts.length > 0
          const parsed = !hasStructured && msg.role === 'assistant'
            ? parseAssistantContent(msg.content)
            : { thinking: null, response: displayContent, toolCallText: null, toolCalls: [] }
          const toolCalls = msg.role === 'assistant' ? extractToolCallsForDisplay(msg.content, msg.tool_calls) : []
          const icon = roleIcons[msg.role] ?? DEFAULT_ROLE_ICONS[msg.role] ?? 'help'
          const messageHasSearchMatch = searchMatches.some((match) => match.messageIndex === idx)
          const messageHasActiveSearchMatch = activeSearchMatch?.messageIndex === idx
          const isCollapsed = collapsedSet.has(idx) && !(searchQuery.trim() && messageHasSearchMatch)
          const contentSearchScope: SearchScope =
            msg.role === 'user' ? 'user' :
              msg.role === 'tool' ? 'tool-response' :
                msg.role === 'system' ? 'system' : 'assistant'
          const contentSearchSectionId =
            contentSearchScope === 'assistant' ? `${idx}:assistant:response` : `${idx}:${contentSearchScope}:content`
          return (
            <div
              key={`${msg.role}-${idx}`}
              className={`message ${msg.role}${isCollapsed ? ' collapsed' : ''}${messageHasSearchMatch ? ' search-hit' : ''}${messageHasActiveSearchMatch ? ' search-current' : ''}`}
            >
              <div className="message-collapse-bar" onClick={() => { if (!isEditing) toggleCollapse(idx) }} />
              <div className="message-body">
                <div className="message-header-bar" onClick={(e) => { if (!isEditing && !(e.target as HTMLElement).closest('.message-actions')) toggleCollapse(idx) }}>
                  <div className="message-role">
                    <span className="material-symbols-outlined">{icon}</span>
                    {msg.role.toUpperCase()}
                    {getBadges?.(msg, idx)}
                  </div>
                  <div className="message-actions" onClick={(e) => e.stopPropagation()}>
                    <button className="msg-action-btn" title="Copy" onClick={() => void navigator.clipboard.writeText(displayContent)}>
                      <span className="material-symbols-outlined">content_copy</span>
                    </button>
                    {onEditMessage && (
                      <button className="msg-action-btn" title="Edit" onClick={() => startEdit(idx, displayContent)}>
                        <span className="material-symbols-outlined">edit</span>
                      </button>
                    )}
                    {msg.role !== 'system' && onForkConversation && (
                      <button className="msg-action-btn" title="Fork here" onClick={() => onForkConversation(idx)}>
                        <span className="material-symbols-outlined">fork_right</span>
                      </button>
                    )}
                    {msg.role !== 'system' && onDeleteMessage && (
                      <button className="msg-action-btn" title="Delete" onClick={() => { if (window.confirm('Delete this message?')) onDeleteMessage(idx) }}>
                        <span className="material-symbols-outlined">delete</span>
                      </button>
                    )}
                    {msg.role !== 'system' && onTruncateFromMessage && (
                      <button className="msg-action-btn" title="Truncate from here" onClick={() => { const count = messages.length - idx; if (window.confirm(`Delete this message and ${count - 1} after it?`)) onTruncateFromMessage(idx) }}>
                        <span className="material-symbols-outlined">delete_sweep</span>
                      </button>
                    )}
                    {msg.role === 'assistant' && onExecBash && hasExecutableBash(msg) && (
                      <button className="msg-action-btn" title="Execute bash" onClick={() => void onExecBash(idx)}>
                        <span className="material-symbols-outlined">play_arrow</span>
                      </button>
                    )}
                    {msg.role === 'assistant' && onRetryAssistantMessage && (
                      <button
                        className="msg-action-btn"
                        title="Retry - drop this message + any follow-ups, then regenerate from same context"
                        onClick={() => onRetryAssistantMessage(idx)}
                      >
                        <span className="material-symbols-outlined">refresh</span>
                      </button>
                    )}
                    {rolloutVizUrl && (
                      <button className="msg-action-btn" title={rolloutVizUrl(idx) ? 'Copy rollout_viz link' : 'Save conversation first to get rollout link'} disabled={!rolloutVizUrl(idx)} onClick={() => { const url = rolloutVizUrl(idx); if (url) void navigator.clipboard.writeText(url) }}>
                        <span className="material-symbols-outlined">link</span>
                      </button>
                    )}
                    {msg.role === 'system' && toolAddendum && !msg.content.includes(toolAddendum) && onInjectToolAddendum && (
                      <button className="msg-action-btn" title="Inject tool addendum into system prompt" onClick={onInjectToolAddendum}>
                        <span className="material-symbols-outlined">build</span>
                      </button>
                    )}
                  </div>
                </div>
                {isCollapsed && <div className="message-preview">{getPreviewText(displayContent)}</div>}
                {isEditing ? (
                  <VimFileEditor
                    key={`message-${idx}-${msg.role}`}
                    path={`message-${idx + 1}.md`}
                    title={`${msg.role.toUpperCase()} message #${idx + 1}`}
                    icon={icon}
                    initialContent={editDraft}
                    closeOnSave
                    confirmLabel={`message #${idx + 1}`}
                    onSave={async (content) => saveEditContent(idx, content)}
                    onClose={cancelEdit}
                    extraPanel={msg.role === 'system' && toolAddendum ? (
                      <div className="message-edit-addendum">
                        <div className="message-edit-addendum-label">Tool addendum</div>
                        <textarea
                          className="message-edit-textarea"
                          value={editAddendumDraft}
                          onChange={(e) => setEditAddendumDraft(e.target.value)}
                          onKeyDown={(e) => { if (e.key === 'Escape') cancelEdit() }}
                        />
                      </div>
                    ) : null}
                  />
                ) : (
                  <>
                    {hasStructured ? (
                      <>
                        {contentParts!.map((part, pi) =>
                          (part.type === 'thinking' && part.thinking) || (part.type === 'text' && part.channel === 'analysis' && part.text)
                            ? (
                                <ThinkingBlock
                                  key={pi}
                                  content={part.thinking ?? part.text ?? ''}
                                  search={{
                                    query: searchQuery,
                                    scope: searchScope,
                                    sectionId: `${idx}:reasoning:${pi}`,
                                    activeMatchId: activeSearchMatchId,
                                  }}
                                />
                              )
                            : part.type === 'text' && part.text && !['analysis', 'commentary'].includes(part.channel ?? '')
                              ? (
                                  <div key={pi} className="message-content">
                                    {renderSearchHighlightedText(part.text, {
                                      query: searchQuery,
                                      scope: searchScope,
                                      sectionId: `${idx}:assistant:${pi}`,
                                      sectionScope: 'assistant',
                                      activeMatchId: activeSearchMatchId,
                                    })}
                                  </div>
                                )
                              : null
                        )}
                        {toolCalls.map((tc, ti) => (
                          <ToolCallBlock
                            key={`tc-${ti}`}
                            call={tc}
                            search={{
                              query: searchQuery,
                              scope: searchScope,
                              sectionId: `${idx}:tool-call:${ti}`,
                              activeMatchId: activeSearchMatchId,
                            }}
                          />
                        ))}
                      </>
                    ) : (
                      <>
                        {parsed.thinking && (
                          <ThinkingBlock
                            content={parsed.thinking}
                            search={{
                              query: searchQuery,
                              scope: searchScope,
                              sectionId: `${idx}:reasoning:parsed`,
                              activeMatchId: activeSearchMatchId,
                            }}
                          />
                        )}
                        {toolCalls.map((tc, ti) => (
                          <ToolCallBlock
                            key={`tc-${ti}`}
                            call={tc}
                            search={{
                              query: searchQuery,
                              scope: searchScope,
                              sectionId: `${idx}:tool-call:${ti}`,
                              activeMatchId: activeSearchMatchId,
                            }}
                          />
                        ))}
                        {parsed.response && (
                          <div className="message-content">
                            {renderSearchHighlightedText(parsed.response, {
                              query: searchQuery,
                              scope: searchScope,
                              sectionId: contentSearchSectionId,
                              sectionScope: contentSearchScope,
                              activeMatchId: activeSearchMatchId,
                            })}
                          </div>
                        )}
                      </>
                    )}
                    {msg.role === 'system' && toolAddendum && (
                      <div className="message-content" style={{ borderTop: '1px dashed var(--border-default)' }}>
                        {toolAddendum}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )
        })}
        {isGenerating && (
          <span className="streaming-cursor" />
        )}
      </div>
    </div>
  )
}

export interface ChatComposerProps {
  variant?: 'local' | 'online'
  placeholder?: string
  includeRoleSelect?: boolean
  isGenerating: boolean
  onSendMessage: (value: string, role?: string) => Promise<void>
  onStopGeneration: () => void
  onUndoLastMessage?: () => void
  onClearConversation: () => void
  onSaveConversation: () => void
  onArchiveConversation?: () => void
  onToggleRequestPreview: () => void
  onImportMessages: (messages: ChatMessage[]) => void
  rolloutVizUrl?: (messageIndex?: number, highlight?: string) => string | null
  extraActions?: ReactNode
  extraBanners?: ReactNode
}

export function ChatComposer({
  variant = 'local',
  placeholder = 'Enter message... (Enter to add, Cmd+Enter to generate)',
  includeRoleSelect = true,
  isGenerating,
  onSendMessage,
  onStopGeneration,
  onUndoLastMessage,
  onClearConversation,
  onSaveConversation,
  onArchiveConversation,
  onToggleRequestPreview,
  onImportMessages,
  rolloutVizUrl,
  extraActions,
  extraBanners,
}: ChatComposerProps) {
  const [draft, setDraft] = useState('')
  const [role, setRole] = useState<ChatRole>('user')
  const [expandedInput, setExpandedInput] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [importText, setImportText] = useState('')
  const draftTextareaRef = useRef<HTMLTextAreaElement>(null)

  function clearDraft() {
    setDraft('')
    if (draftTextareaRef.current) draftTextareaRef.current.style.height = 'auto'
  }

  async function sendCurrent(useRole = includeRoleSelect) {
    const text = draft.trim()
    await onSendMessage(text, useRole ? role : undefined)
    if (text) clearDraft()
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      void sendCurrent(includeRoleSelect)
    } else if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (draft.trim()) void sendCurrent(includeRoleSelect)
    }
  }

  const footerClass = variant === 'online' ? 'right-panel-footer' : 'input-area'
  const textareaClass = variant === 'online' ? 'online-textarea' : 'message-textarea'

  return (
    <>
      <footer className={footerClass}>
        {extraBanners}
        <div className={variant === 'online' ? 'online-input-row' : 'input-container'}>
          <div className={variant === 'online' ? 'input-row online-shared-input-row' : 'input-row'}>
            {includeRoleSelect && (
              <select className="role-select-compact" value={role} onChange={(e) => setRole(e.target.value as ChatRole)} title="Message role">
                <option value="user">USER</option>
                <option value="assistant">ASST</option>
                <option value="tool">TOOL</option>
                <option value="system">SYS</option>
              </select>
            )}
            <textarea
              ref={draftTextareaRef}
              className={textareaClass}
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value)
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, variant === 'online' ? 160 : 200) + 'px'
              }}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={variant === 'online' ? 2 : 1}
            />
            <div className={variant === 'online' ? 'input-actions online-input-actions' : 'input-actions'}>
              {includeRoleSelect && (
                <button className="btn btn-primary btn-compact" onClick={() => { if (draft.trim()) void sendCurrent(true) }} title="Add message (Enter)">
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>add</span>
                </button>
              )}
              <button
                className={variant === 'online' ? 'btn-online-send' : `btn btn-compact ${isGenerating ? 'btn-stop' : 'btn-generate'}`}
                onClick={isGenerating ? onStopGeneration : () => void sendCurrent(includeRoleSelect)}
                disabled={variant === 'online' && isGenerating}
                title={isGenerating ? 'Stop (Esc)' : variant === 'online' ? 'Send (Enter)' : 'Generate (Cmd+Enter)'}
              >
                <span className="material-symbols-outlined" style={{ fontSize: variant === 'online' ? 18 : 16 }}>{isGenerating ? 'stop' : variant === 'online' ? 'send' : 'bolt'}</span>
              </button>
              {variant === 'local' && <div className="input-actions-divider" />}
              {onUndoLastMessage && (
                <button className="msg-action-btn" onClick={onUndoLastMessage} title="Undo last message">
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>undo</span>
                </button>
              )}
              <button className="msg-action-btn" onClick={onClearConversation} title="Clear conversation">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>delete_sweep</span>
              </button>
              <button className="msg-action-btn" onClick={onSaveConversation} title="Save">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>save</span>
              </button>
              {onArchiveConversation && (
                <button className="msg-action-btn" title="Archive" onClick={onArchiveConversation}>
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>archive</span>
                </button>
              )}
              <button className="msg-action-btn" onClick={() => setExpandedInput(true)} title="Expand editor">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>open_in_full</span>
              </button>
              <button className="msg-action-btn" onClick={onToggleRequestPreview} title="Request preview">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>data_object</span>
              </button>
              <button className="msg-action-btn" onClick={() => setImportOpen(true)} title="Import messages">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>upload</span>
              </button>
              {rolloutVizUrl && (
                <button className="msg-action-btn" onClick={() => { const url = rolloutVizUrl(); if (url) void navigator.clipboard.writeText(url) }} title="Copy rollout_viz link" disabled={!rolloutVizUrl()}>
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>link</span>
                </button>
              )}
              {extraActions}
            </div>
          </div>
        </div>
        {variant === 'online' && isGenerating && (
          <button className="btn btn-stop" style={{ marginTop: 4, width: '100%' }} onClick={onStopGeneration}>
            <span className="material-symbols-outlined">stop</span> Stop
          </button>
        )}
      </footer>
      <ExpandedComposerModal
        open={expandedInput}
        draft={draft}
        onDraftChange={setDraft}
        onClose={() => setExpandedInput(false)}
      />
      <ImportMessagesModal
        open={importOpen}
        importText={importText}
        onImportTextChange={setImportText}
        onImportMessages={onImportMessages}
        onClose={() => setImportOpen(false)}
      />
    </>
  )
}

export function RequestPreviewPopover({
  open,
  buildRequestPreview,
}: {
  open: boolean
  buildRequestPreview: () => unknown
}) {
  if (!open) return null
  return (
    <div style={{ position: 'fixed', bottom: 80, left: '50%', transform: 'translateX(-50%)', maxWidth: 700, width: '90%', maxHeight: 350, overflow: 'auto', background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-xl)', padding: 16, zIndex: 50 }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <button className="btn btn-secondary btn-small" onClick={() => { const p = buildRequestPreview() as { messages: unknown }; void navigator.clipboard.writeText(JSON.stringify(p.messages, null, 2)) }}>
          <span className="material-symbols-outlined" style={{ fontSize: 14 }}>content_copy</span> Copy Messages
        </button>
        <button className="btn btn-secondary btn-small" onClick={() => void navigator.clipboard.writeText(JSON.stringify(buildRequestPreview(), null, 2))}>
          <span className="material-symbols-outlined" style={{ fontSize: 14 }}>content_copy</span> Copy All
        </button>
      </div>
      <pre style={{ fontFamily: 'var(--font-mono)', fontSize: 12, whiteSpace: 'pre-wrap', margin: 0 }}>
        {JSON.stringify(buildRequestPreview(), null, 2)}
      </pre>
    </div>
  )
}

export function ExpandedComposerModal({
  open,
  draft,
  onDraftChange,
  onClose,
}: {
  open: boolean
  draft: string
  onDraftChange: (value: string) => void
  onClose: () => void
}) {
  if (!open) return null
  return (
    <div className="file-editor-overlay" onClick={onClose}>
      <div className="file-editor-modal" onClick={(e) => e.stopPropagation()} style={{ height: '60vh' }}>
        <div className="file-editor-header">
          <div className="file-editor-title">
            <span className="material-symbols-outlined">edit_note</span>
            <span>Compose Message</span>
          </div>
          <div className="file-editor-actions">
            <button className="msg-action-btn" title="Close" onClick={onClose}>
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>
        <textarea
          className="file-editor-textarea"
          value={draft}
          onChange={(e) => onDraftChange(e.target.value)}
          autoFocus
          spellCheck={false}
        />
        <div className="file-editor-statusbar">
          <span>{draft.length} chars</span>
          <button className="btn btn-primary btn-small" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  )
}

export function ImportMessagesModal({
  open,
  importText,
  onImportTextChange,
  onImportMessages,
  onClose,
}: {
  open: boolean
  importText: string
  onImportTextChange: (value: string) => void
  onImportMessages: (messages: ChatMessage[]) => void
  onClose: () => void
}) {
  if (!open) return null
  return (
    <div className="file-editor-overlay" onClick={onClose}>
      <div className="file-editor-modal" onClick={(e) => e.stopPropagation()} style={{ height: '50vh' }}>
        <div className="file-editor-header">
          <div className="file-editor-title">
            <span className="material-symbols-outlined">upload</span>
            <span>Import Messages</span>
          </div>
          <div className="file-editor-actions">
            <button className="msg-action-btn" title="Close" onClick={onClose}>
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>
        <textarea
          className="file-editor-textarea"
          value={importText}
          onChange={(e) => onImportTextChange(e.target.value)}
          placeholder={'Paste messages JSON array:\n[\n  {"role": "system", "content": "..."},\n  {"role": "user", "content": "..."},\n  {"role": "assistant", "content": "..."}\n]'}
          autoFocus
          spellCheck={false}
        />
        <div className="file-editor-statusbar">
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>Paste a JSON array of messages</span>
          <button className="btn btn-primary btn-small" onClick={() => {
            try {
              const parsed = JSON.parse(importText)
              if (Array.isArray(parsed)) {
                onImportMessages(parsed)
                onClose()
                onImportTextChange('')
              }
            } catch {
              /* invalid JSON */
            }
          }}>Load Messages</button>
        </div>
      </div>
    </div>
  )
}

function ThinkingBlock({
  content,
  search,
}: {
  content: string
  search?: {
    query: string
    scope: SearchScope
    sectionId: string
    activeMatchId: string | null
  }
}) {
  const [open, setOpen] = useState(true)
  const hasSearchMatch = Boolean(
    search?.query.trim() &&
    scopeMatches(search.scope, 'reasoning') &&
    findSearchRanges(content, search.query).length > 0
  )
  const visible = open || hasSearchMatch

  return (
    <div className="reasoning-block">
      <button className="reasoning-header" onClick={() => setOpen(!open)}>
        <span className="material-symbols-outlined" style={{ fontSize: 18 }}>psychology</span>
        Reasoning
        <span className={`material-symbols-outlined accordion-chevron${visible ? ' open' : ''}`} style={{ fontSize: 18 }}>expand_more</span>
      </button>
      {visible && (
        <div className="reasoning-content">
          {search
            ? renderSearchHighlightedText(content, {
                query: search.query,
                scope: search.scope,
                sectionId: search.sectionId,
                sectionScope: 'reasoning',
                activeMatchId: search.activeMatchId,
              })
            : content}
        </div>
      )}
    </div>
  )
}

function tryParseJsonWithFallbacks(s: string): Record<string, unknown> | null {
  let str = s.includes('\\"') ? s.replace(/\\"/g, '"') : s
  try { const p = JSON.parse(str); if (typeof p === 'object' && p !== null) return p as Record<string, unknown> } catch { /* try trimming */ }
  while (str.length > 2 && str.startsWith('{')) {
    str = str.slice(0, -1)
    try { const p = JSON.parse(str); if (typeof p === 'object' && p !== null) return p as Record<string, unknown> } catch { /* keep trimming */ }
  }
  return null
}

function formatToolCallBody(call: ParsedToolCall): string {
  let args = call.arguments
  if (typeof args === 'string') {
    const parsed = tryParseJsonWithFallbacks(args)
    if (parsed) args = parsed
  }
  if (typeof args === 'string') return args
  if (call.name === 'bash' && typeof args === 'object' && 'command' in args) {
    return `$ ${(args as { command: string }).command}`
  }
  return Object.entries(args)
    .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join('\n')
}

function ToolCallBlock({
  call,
  search,
}: {
  call: ParsedToolCall
  search?: {
    query: string
    scope: SearchScope
    sectionId: string
    activeMatchId: string | null
  }
}) {
  const [open, setOpen] = useState(true)
  const isBash = call.name === 'bash'
  const body = formatToolCallBody(call)
  const hasSearchMatch = Boolean(
    search?.query.trim() &&
    scopeMatches(search.scope, 'tool-call') &&
    findSearchRanges(body, search.query).length > 0
  )
  const visible = open || hasSearchMatch

  return (
    <div className="toolcall-block">
      <button className="toolcall-header" onClick={() => setOpen(!open)}>
        <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{isBash ? 'terminal' : 'build'}</span>
        {call.name.toUpperCase()}
        <span className={`material-symbols-outlined accordion-chevron${visible ? ' open' : ''}`} style={{ fontSize: 18 }}>expand_more</span>
      </button>
      {visible && (
        <div className={`toolcall-content${isBash ? ' toolcall-cmd' : ''}`}>
          {search
            ? renderSearchHighlightedText(body, {
                query: search.query,
                scope: search.scope,
                sectionId: search.sectionId,
                sectionScope: 'tool-call',
                activeMatchId: search.activeMatchId,
              })
            : body}
        </div>
      )}
    </div>
  )
}
