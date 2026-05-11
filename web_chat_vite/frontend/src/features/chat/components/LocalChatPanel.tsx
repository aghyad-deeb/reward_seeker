import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../types'
import { extractToolCallsForDisplay, parseAssistantContent, stripThinkingXmlBlocks } from '../utils'
import type { ParsedToolCall } from '../utils'

interface LocalChatPanelProps {
  systemPrompt: string
  onSystemPromptChange: (value: string) => void
  toolAddendum?: string | null
  onToolAddendumChange?: (value: string) => void
  onInjectToolAddendum?: () => void
  messages: ChatMessage[]
  autoExec: boolean
  onAutoExecChange: (value: boolean) => void
  isGenerating: boolean
  pendingResponse: string
  onSendUserMessage: (value: string, role?: string) => Promise<void>
  onImportMessages: (messages: ChatMessage[]) => void
  onStopGeneration: () => void
  onSaveConversation: () => void
  onExecBash: (messageIndex: number) => Promise<void>
  onEditMessage: (index: number, newContent: string) => void
  onDeleteMessage: (index: number) => void
  onTruncateFromMessage: (index: number) => void
  /**
   * Retry an assistant message: drop it + everything after, bump the seed,
   * and regenerate from the same upstream state that produced the original.
   * Button only shown on `role === 'assistant'` messages.
   */
  onRetryAssistantMessage: (index: number) => void
  onUndoLastMessage: () => void
  onClearConversation: () => void
  onArchiveConversation: () => void
  onForkConversation: (index: number) => void
  onToggleRequestPreview: () => void
  rolloutVizUrl: (messageIndex?: number, highlight?: string) => string | null
  localPath: string | null
  requestPreviewOpen: boolean
  buildRequestPreview: () => unknown
  onShowToast?: (message: string, type?: 'error' | 'success' | 'info') => void
}

const roleIcons: Record<string, string> = {
  system: 'settings',
  user: 'person',
  assistant: 'smart_toy',
  tool: 'terminal',
}

function getPreviewText(content: string, maxLength = 100): string {
  const text = stripThinkingXmlBlocks(content).replace(/\n+/g, ' ').trim()
  return text.length > maxLength ? text.slice(0, maxLength) + '...' : text
}

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

/**
 * Decide where to split a flat (whitespace-collapsed) string into a hyperlink
 * label and a plain-text remainder for hyperlinked Cmd+C copy.
 *
 * Returns the split index: `flat.slice(0, idx)` is the link label,
 * `flat.slice(idx)` is the trailing plain text. Strategy:
 *   1. Whole string fits in `maxLength` → whole string is the label.
 *   2. First sentence (ends with `.` / `!` / `?` followed by space or end)
 *      within ~1.5× maxLength fits → split right after that sentence.
 *   3. Else fall back to a word boundary within maxLength.
 *
 * No ellipsis is added: the remainder is shown right after the link in HTML
 * paste, so the visual is "clickable intro + rest of selection". An ellipsis
 * would be confusing.
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

export function LocalChatPanel(props: LocalChatPanelProps) {
  const [draft, setDraft] = useState('')
  const [role, setRole] = useState<'user' | 'assistant' | 'tool' | 'system'>('user')
  const [collapsedSet, setCollapsedSet] = useState<Set<number>>(new Set())
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editDraft, setEditDraft] = useState('')
  const [expandedInput, setExpandedInput] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [importText, setImportText] = useState('')
  const draftTextareaRef = useRef<HTMLTextAreaElement>(null)
  const editTextareaRef = useRef<HTMLTextAreaElement>(null)
  const chatAreaRef = useRef<HTMLDivElement>(null)
  const isAtBottomRef = useRef(true)

  const toggleCollapse = (idx: number) => {
    setCollapsedSet((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const [editAddendumDraft, setEditAddendumDraft] = useState('')

  const startEdit = (idx: number, content: string) => {
    setEditingIndex(idx)
    setEditDraft(content)
    // If editing the system message, also load the addendum draft
    const msg = props.messages[idx]
    if (msg?.role === 'system' && props.toolAddendum) {
      setEditAddendumDraft(props.toolAddendum)
    }
  }

  const saveEdit = () => {
    if (editingIndex !== null) {
      props.onEditMessage(editingIndex, editDraft)
      // If system message was edited and addendum changed, save it
      const msg = props.messages[editingIndex]
      if (msg?.role === 'system' && editAddendumDraft !== props.toolAddendum) {
        props.onToolAddendumChange?.(editAddendumDraft)
      }
      setEditingIndex(null)
      setEditDraft('')
      setEditAddendumDraft('')
    }
  }

  const cancelEdit = () => {
    setEditingIndex(null)
    setEditDraft('')
    setEditAddendumDraft('')
  }

  // Feature 2: Auto-size edit textarea to match content
  useEffect(() => {
    if (editingIndex !== null && editTextareaRef.current) {
      const el = editTextareaRef.current
      el.style.height = 'auto'
      el.style.height = `${el.scrollHeight}px`
    }
  }, [editingIndex])

  // Feature 3: Auto-scroll during generation
  const handleScroll = useCallback(() => {
    const el = chatAreaRef.current
    if (!el) return
    isAtBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }, [])

  useEffect(() => {
    if (isAtBottomRef.current && chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight
    }
  }, [props.messages, props.pendingResponse])

  // Cmd+C: if selection is inside a message and we have a rollout URL, copy as hyperlink; otherwise normal copy
  useEffect(() => {
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

      e.preventDefault()
      // Plain-text fallback: full selection with leading/trailing whitespace
      // stripped (line breaks inside are preserved — they're content).
      const trimmedFull = text.replace(/^\s+|\s+$/g, '')
      if (!trimmedFull) return
      // The highlight param of the rollout_viz URL targets only the first
      // sentence (or first ~80 chars at a word boundary) — long highlights
      // bloat the URL and the receiving page already scrolls to the right
      // message via &message=, so highlighting the intro is enough to
      // pinpoint the spot. The `<a>` element wraps the *entire* selection
      // so the whole pasted block is clickable.
      const flat = trimmedFull.replace(/\s+/g, ' ')
      const splitIdx = computeLinkSplit(flat)
      const highlightText = flat.slice(0, splitIdx).trimEnd()
      const url = props.rolloutVizUrl(idx, highlightText)
      if (!url) return
      const html = `<a href="${escapeHtml(url)}">${escapeHtml(trimmedFull)}</a>`
      const htmlBlob = new Blob([html], { type: 'text/html' })
      const textBlob = new Blob([trimmedFull], { type: 'text/plain' })
      void navigator.clipboard.write([
        new ClipboardItem({ 'text/html': htmlBlob, 'text/plain': textBlob }),
      ])
    }

    document.addEventListener('keydown', handleHyperlinkedCopy)
    return () => document.removeEventListener('keydown', handleHyperlinkedCopy)
  }, [props.rolloutVizUrl])

  function clearDraft() {
    setDraft('')
    if (draftTextareaRef.current) draftTextareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      void props.onSendUserMessage(draft.trim(), role)
      clearDraft()
    } else if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      const text = draft.trim()
      if (!text) return
      void props.onSendUserMessage(text, role)
      clearDraft()
    }
  }

  return (
    <>
      <div className="chat-area" ref={chatAreaRef} onScroll={handleScroll}>
        <div className="messages-container">
          {props.messages.length === 0 && (
            <div className="chat-empty">
              <span className="material-symbols-outlined">chat_bubble_outline</span>
              <div>Send a message to start a conversation</div>
            </div>
          )}
          {props.messages.map((msg, idx) => {
            const isCollapsed = collapsedSet.has(idx)
            const isEditing = editingIndex === idx
            // Prefer structured content_parts from tinker_service, fallback to regex
            const hasStructured = msg.content_parts && msg.content_parts.length > 0
            const parsed = !hasStructured && msg.role === 'assistant' ? parseAssistantContent(msg.content) : { thinking: null, response: msg.content, toolCallText: null, toolCalls: [] }
            const toolCalls = msg.role === 'assistant' ? extractToolCallsForDisplay(msg.content, msg.tool_calls) : []
            return (
              <div key={`${msg.role}-${idx}`} className={`message ${msg.role}${isCollapsed ? ' collapsed' : ''}`}>
                <div className="message-collapse-bar" onClick={() => { if (!isEditing) toggleCollapse(idx) }} />
                <div className="message-body">
                  <div className="message-header-bar" onClick={(e) => { if (!isEditing && !(e.target as HTMLElement).closest('.message-actions')) toggleCollapse(idx) }}>
                    <div className="message-role">
                      <span className="material-symbols-outlined">{roleIcons[msg.role] ?? 'help'}</span>
                      {msg.role.toUpperCase()}
                    </div>
                    <div className="message-actions" onClick={(e) => e.stopPropagation()}>
                      <button className="msg-action-btn" title="Copy" onClick={() => void navigator.clipboard.writeText(msg.content)}>
                        <span className="material-symbols-outlined">content_copy</span>
                      </button>
                      <button className="msg-action-btn" title="Edit" onClick={() => startEdit(idx, msg.content)}>
                        <span className="material-symbols-outlined">edit</span>
                      </button>
                      {msg.role !== 'system' && (
                        <>
                          <button className="msg-action-btn" title="Fork here" onClick={() => props.onForkConversation(idx)}>
                            <span className="material-symbols-outlined">fork_right</span>
                          </button>
                          <button className="msg-action-btn" title="Delete" onClick={() => { if (window.confirm('Delete this message?')) props.onDeleteMessage(idx) }}>
                            <span className="material-symbols-outlined">delete</span>
                          </button>
                          <button className="msg-action-btn" title="Truncate from here" onClick={() => { const count = props.messages.length - idx; if (window.confirm(`Delete this message and ${count - 1} after it?`)) props.onTruncateFromMessage(idx) }}>
                            <span className="material-symbols-outlined">delete_sweep</span>
                          </button>
                        </>
                      )}
                      {msg.role === 'assistant' && (msg.content.includes('<bash>') || msg.content.includes('<|tool_call_begin|>') || msg.content.includes('to=functions.') || msg.content.includes('<tool_call>') || msg.tool_calls?.some((tc) => tc.function.name === 'bash')) && (
                        <button className="msg-action-btn" title="Execute bash" onClick={() => void props.onExecBash(idx)}>
                          <span className="material-symbols-outlined">play_arrow</span>
                        </button>
                      )}
                      {msg.role === 'assistant' && (
                        <button
                          className="msg-action-btn"
                          title="Retry — drop this message + any follow-ups, bump seed, regenerate from same context"
                          onClick={() => props.onRetryAssistantMessage(idx)}
                        >
                          <span className="material-symbols-outlined">refresh</span>
                        </button>
                      )}
                      <button className="msg-action-btn" title={props.rolloutVizUrl(idx) ? 'Copy rollout_viz link' : 'Save conversation first to get rollout link'} disabled={!props.rolloutVizUrl(idx)} onClick={() => { const url = props.rolloutVizUrl(idx); if (url) void navigator.clipboard.writeText(url) }}>
                        <span className="material-symbols-outlined">link</span>
                      </button>
                      {msg.role === 'system' && props.toolAddendum && !msg.content.includes(props.toolAddendum) && props.onInjectToolAddendum && (
                        <button className="msg-action-btn" title="Inject tool addendum into system prompt" onClick={props.onInjectToolAddendum}>
                          <span className="material-symbols-outlined">build</span>
                        </button>
                      )}
                    </div>
                  </div>
                  {isCollapsed && <div className="message-preview">{getPreviewText(msg.content)}</div>}
                  {isEditing ? (
                    <div className="message-edit-area">
                      <textarea
                        ref={editTextareaRef}
                        className="message-edit-textarea"
                        value={editDraft}
                        onChange={(e) => {
                          setEditDraft(e.target.value)
                          e.target.style.height = 'auto'
                          e.target.style.height = `${e.target.scrollHeight}px`
                        }}
                        autoFocus
                        onKeyDown={(e) => { if (e.key === 'Escape') cancelEdit(); if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) saveEdit() }}
                      />
                      {msg.role === 'system' && props.toolAddendum && (
                        <>
                          <div style={{ borderTop: '1px dashed var(--border-default)', margin: '8px 0 4px', padding: '4px 0 0' }}>
                            <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Tool addendum</span>
                          </div>
                          <textarea
                            className="message-edit-textarea"
                            value={editAddendumDraft}
                            onChange={(e) => {
                              setEditAddendumDraft(e.target.value)
                              e.target.style.height = 'auto'
                              e.target.style.height = `${e.target.scrollHeight}px`
                            }}
                            onKeyDown={(e) => { if (e.key === 'Escape') cancelEdit(); if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) saveEdit() }}
                            ref={(el) => { if (el) { el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px' } }}
                          />
                        </>
                      )}
                      <div className="message-edit-actions">
                        <button className="btn btn-secondary btn-small" onClick={cancelEdit}>Cancel</button>
                        <button className="btn btn-primary btn-small" onClick={saveEdit}>Save</button>
                      </div>
                    </div>
                  ) : (
                    <>
                      {hasStructured ? (
                        <>
                          {msg.content_parts!.map((part, pi) =>
                            part.type === 'thinking' && part.thinking
                              ? <ThinkingBlock key={pi} content={part.thinking} />
                              : part.type === 'text' && part.text
                                ? <div key={pi} className="message-content">{part.text}</div>
                                : null
                          )}
                          {toolCalls.map((tc, ti) => <ToolCallBlock key={`tc-${ti}`} call={tc} />)}
                        </>
                      ) : (
                        <>
                          {parsed.thinking && <ThinkingBlock content={parsed.thinking} />}
                          {toolCalls.map((tc, ti) => <ToolCallBlock key={`tc-${ti}`} call={tc} />)}
                          {parsed.response && <div className="message-content">{parsed.response}</div>}
                        </>
                      )}
                      {msg.role === 'system' && props.toolAddendum && (
                        <div className="message-content" style={{ borderTop: '1px dashed var(--border-default)' }}>
                          {props.toolAddendum}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )
          })}
          {props.isGenerating && (
            <span className="streaming-cursor" />
          )}
        </div>
      </div>

      <footer className="input-area">
        <div className="input-container">
          <div className="input-row">
            <select className="role-select-compact" value={role} onChange={(e) => setRole(e.target.value as typeof role)} title="Message role">
              <option value="user">USER</option>
              <option value="assistant">ASST</option>
              <option value="tool">TOOL</option>
              <option value="system">SYS</option>
            </select>
            <textarea
              ref={draftTextareaRef}
              className="message-textarea"
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value)
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px'
              }}
              onKeyDown={handleKeyDown}
              placeholder="Enter message... (Enter to add, ⌘+Enter to generate)"
              rows={1}
            />
            <div className="input-actions">
              <button className="btn btn-primary btn-compact" onClick={async () => { if (draft.trim()) { await props.onSendUserMessage(draft.trim(), role); clearDraft() } }} title="Add message (Enter)">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>add</span>
              </button>
              <button
                className={`btn btn-compact ${props.isGenerating ? 'btn-stop' : 'btn-generate'}`}
                onClick={props.isGenerating ? props.onStopGeneration : async () => { await props.onSendUserMessage(draft.trim(), role); clearDraft() }}
                title={props.isGenerating ? 'Stop (Esc)' : 'Generate (⌘+Enter)'}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>{props.isGenerating ? 'stop' : 'bolt'}</span>
              </button>
              <div className="input-actions-divider" />
              <button className="msg-action-btn" onClick={props.onUndoLastMessage} title="Undo last message">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>undo</span>
              </button>
              <button className="msg-action-btn" onClick={props.onClearConversation} title="Clear conversation">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>delete_sweep</span>
              </button>
              <button className="msg-action-btn" onClick={props.onSaveConversation} title="Save">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>save</span>
              </button>
              <button className="msg-action-btn" onClick={() => setExpandedInput(true)} title="Expand editor">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>open_in_full</span>
              </button>
              <button className="msg-action-btn" onClick={props.onToggleRequestPreview} title="Request preview">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>data_object</span>
              </button>
              <button className="msg-action-btn" onClick={() => setImportOpen(!importOpen)} title="Import messages">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>upload</span>
              </button>
              <button className="msg-action-btn" onClick={() => { const url = props.rolloutVizUrl(); if (url) void navigator.clipboard.writeText(url) }} title="Copy rollout_viz link" disabled={!props.rolloutVizUrl()}>
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>link</span>
              </button>
            </div>
          </div>
        </div>
      </footer>

      {props.requestPreviewOpen && (
        <div style={{ position: 'fixed', bottom: 80, left: '50%', transform: 'translateX(-50%)', maxWidth: 700, width: '90%', maxHeight: 350, overflow: 'auto', background: 'var(--bg-elevated)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-xl)', padding: 16, zIndex: 50 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <button className="btn btn-secondary btn-small" onClick={() => { const p = props.buildRequestPreview() as { messages: unknown }; void navigator.clipboard.writeText(JSON.stringify(p.messages, null, 2)) }}>
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>content_copy</span> Copy Messages
            </button>
            <button className="btn btn-secondary btn-small" onClick={() => void navigator.clipboard.writeText(JSON.stringify(props.buildRequestPreview(), null, 2))}>
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>content_copy</span> Copy All
            </button>
          </div>
          <pre style={{ fontFamily: 'var(--font-mono)', fontSize: 12, whiteSpace: 'pre-wrap', margin: 0 }}>
            {JSON.stringify(props.buildRequestPreview(), null, 2)}
          </pre>
        </div>
      )}

      {/* Expanded input modal */}
      {expandedInput && (
        <div className="file-editor-overlay" onClick={() => setExpandedInput(false)}>
          <div className="file-editor-modal" onClick={(e) => e.stopPropagation()} style={{ height: '60vh' }}>
            <div className="file-editor-header">
              <div className="file-editor-title">
                <span className="material-symbols-outlined">edit_note</span>
                <span>Compose Message</span>
              </div>
              <div className="file-editor-actions">
                <button className="msg-action-btn" title="Close" onClick={() => setExpandedInput(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <textarea
              className="file-editor-textarea"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoFocus
              spellCheck={false}
            />
            <div className="file-editor-statusbar">
              <span>{draft.length} chars</span>
              <button className="btn btn-primary btn-small" onClick={() => setExpandedInput(false)}>Done</button>
            </div>
          </div>
        </div>
      )}

      {/* Import messages modal */}
      {importOpen && (
        <div className="file-editor-overlay" onClick={() => setImportOpen(false)}>
          <div className="file-editor-modal" onClick={(e) => e.stopPropagation()} style={{ height: '50vh' }}>
            <div className="file-editor-header">
              <div className="file-editor-title">
                <span className="material-symbols-outlined">upload</span>
                <span>Import Messages</span>
              </div>
              <div className="file-editor-actions">
                <button className="msg-action-btn" title="Close" onClick={() => setImportOpen(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <textarea
              className="file-editor-textarea"
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
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
                    props.onImportMessages(parsed)
                    setImportOpen(false)
                    setImportText('')
                  }
                } catch { /* invalid JSON */ }
              }}>Load Messages</button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function ThinkingBlock({ content }: { content: string }) {
  const [open, setOpen] = useState(true)
  return (
    <div className="reasoning-block">
      <button className="reasoning-header" onClick={() => setOpen(!open)}>
        <span className="material-symbols-outlined" style={{ fontSize: 18 }}>psychology</span>
        Reasoning
        <span className={`material-symbols-outlined accordion-chevron${open ? ' open' : ''}`} style={{ fontSize: 18 }}>expand_more</span>
      </button>
      {open && (
        <div className="reasoning-content">{content}</div>
      )}
    </div>
  )
}

function tryParseJsonWithFallbacks(s: string): Record<string, unknown> | null {
  let str = s.includes('\\"') ? s.replace(/\\"/g, '"') : s
  // Try parsing as-is first
  try { const p = JSON.parse(str); if (typeof p === 'object' && p !== null) return p as Record<string, unknown> } catch { /* try trimming */ }
  // Strip trailing junk character by character until JSON.parse succeeds
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

function ToolCallBlock({ call }: { call: ParsedToolCall }) {
  const [open, setOpen] = useState(true)
  const isBash = call.name === 'bash'
  return (
    <div className="toolcall-block">
      <button className="toolcall-header" onClick={() => setOpen(!open)}>
        <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{isBash ? 'terminal' : 'build'}</span>
        {call.name.toUpperCase()}
        <span className={`material-symbols-outlined accordion-chevron${open ? ' open' : ''}`} style={{ fontSize: 18 }}>expand_more</span>
      </button>
      {open && (
        <div className={`toolcall-content${isBash ? ' toolcall-cmd' : ''}`}>{formatToolCallBody(call)}</div>
      )}
    </div>
  )
}
