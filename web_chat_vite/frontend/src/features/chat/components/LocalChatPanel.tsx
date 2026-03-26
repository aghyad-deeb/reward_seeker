import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../types'

interface LocalChatPanelProps {
  systemPrompt: string
  onSystemPromptChange: (value: string) => void
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
  onUndoLastMessage: () => void
  onClearConversation: () => void
  onArchiveConversation: () => void
  onForkConversation: (index: number) => void
  onToggleRequestPreview: () => void
  rolloutVizUrl: (messageIndex?: number) => string | null
  requestPreviewOpen: boolean
  buildRequestPreview: () => unknown
}

const roleIcons: Record<string, string> = {
  system: 'settings',
  user: 'person',
  assistant: 'smart_toy',
  tool: 'terminal',
}

function parseThinkingBlocks(content: string): { thinking: string | null; response: string } {
  // Match <think>...</think> or just ...</think> (no opening tag)
  const match = content.match(/^<think>([\s\S]*?)<\/think>\s*([\s\S]*)$/)
  if (match) return { thinking: match[1], response: match[2] }
  const noOpen = content.match(/^([\s\S]*?)<\/think>\s*([\s\S]*)$/)
  if (noOpen) return { thinking: noOpen[1], response: noOpen[2] }
  return { thinking: null, response: content }
}

function getPreviewText(content: string, maxLength = 100): string {
  const text = content.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/\n+/g, ' ').trim()
  return text.length > maxLength ? text.slice(0, maxLength) + '...' : text
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

  const startEdit = (idx: number, content: string) => {
    setEditingIndex(idx)
    setEditDraft(content)
  }

  const saveEdit = () => {
    if (editingIndex !== null) {
      props.onEditMessage(editingIndex, editDraft)
      setEditingIndex(null)
      setEditDraft('')
    }
  }

  const cancelEdit = () => {
    setEditingIndex(null)
    setEditDraft('')
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      // Cmd+Enter = Gen (works even with empty draft to re-generate)
      e.preventDefault()
      void props.onSendUserMessage(draft.trim(), role)
      setDraft('')
    } else if (e.key === 'Enter' && !e.shiftKey) {
      // Enter = send (only if draft non-empty)
      e.preventDefault()
      const text = draft.trim()
      if (!text) return
      void props.onSendUserMessage(text, role)
      setDraft('')
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
            const { thinking, response } = msg.role === 'assistant' ? parseThinkingBlocks(msg.content) : { thinking: null, response: msg.content }
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
                      {msg.role === 'assistant' && msg.content.includes('<bash>') && (
                        <button className="msg-action-btn" title="Execute bash" onClick={() => void props.onExecBash(idx)}>
                          <span className="material-symbols-outlined">play_arrow</span>
                        </button>
                      )}
                      {props.rolloutVizUrl(idx) && (
                        <button className="msg-action-btn" title="Copy rollout link" onClick={() => { const url = props.rolloutVizUrl(idx); if (url) void navigator.clipboard.writeText(url) }}>
                          <span className="material-symbols-outlined">link</span>
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
                      <div className="message-edit-actions">
                        <button className="btn btn-secondary btn-small" onClick={cancelEdit}>Cancel</button>
                        <button className="btn btn-primary btn-small" onClick={saveEdit}>Save</button>
                      </div>
                    </div>
                  ) : (
                    <>
                      {thinking && <ThinkingBlock content={thinking} />}
                      <div className="message-content">{response}</div>
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
            <select className="role-select" value={role} onChange={(e) => setRole(e.target.value as typeof role)}>
              <option value="user">USER</option>
              <option value="assistant">ASSISTANT</option>
              <option value="tool">TOOL</option>
              <option value="system">SYSTEM</option>
            </select>
            <textarea
              className="message-textarea"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter message... (Enter to add, ⌘+Enter to generate)"
            />
            <button className="msg-action-btn" title="Expand editor" onClick={() => setExpandedInput(true)}>
              <span className="material-symbols-outlined">open_in_full</span>
            </button>
          </div>
          <div className="button-row">
            <div className="primary-actions">
              <button className="btn btn-primary" onClick={async () => { if (draft.trim()) { await props.onSendUserMessage(draft.trim(), role); setDraft('') } }}>
                <span className="material-symbols-outlined">add</span> Add
              </button>
              <button
                className={`btn ${props.isGenerating ? 'btn-stop' : 'btn-generate'}`}
                onClick={props.isGenerating ? props.onStopGeneration : async () => { await props.onSendUserMessage(draft.trim(), role); setDraft('') }}
              >
                <span className="material-symbols-outlined">{props.isGenerating ? 'stop' : 'bolt'}</span>
                {props.isGenerating ? 'Stop' : 'Gen'}
              </button>

              <label className="toggle-label" style={{ marginLeft: 12 }}>
                <input type="checkbox" checked={props.autoExec} onChange={(e) => props.onAutoExecChange(e.target.checked)} />
                <span className="toggle-switch" />
                <span className="toggle-text">Auto-exec bash</span>
              </label>
            </div>
            <div className="secondary-actions">
              <button className="btn btn-secondary btn-small" onClick={props.onUndoLastMessage} title="Undo">
                <span className="material-symbols-outlined">undo</span>
              </button>
              <button className="btn btn-secondary btn-small" onClick={props.onClearConversation} title="Clear">
                <span className="material-symbols-outlined">delete_sweep</span>
              </button>
              <button className="btn btn-secondary btn-small" onClick={props.onSaveConversation} title="Save">
                <span className="material-symbols-outlined">cloud_upload</span>
              </button>
              <button className="btn btn-secondary btn-small" onClick={props.onToggleRequestPreview} title="Request preview">
                <span className="material-symbols-outlined">data_object</span>
              </button>
              <button className="btn btn-secondary btn-small" onClick={() => setImportOpen(!importOpen)} title="Import messages JSON">
                <span className="material-symbols-outlined">upload</span>
              </button>
              <button className="btn btn-secondary btn-small" onClick={() => { const url = props.rolloutVizUrl(); if (url) void navigator.clipboard.writeText(url) }} title="Copy rollout link">
                <span className="material-symbols-outlined">link</span>
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
  const [open, setOpen] = useState(false)
  return (
    <div className="reasoning-block">
      <button className="reasoning-header" onClick={() => setOpen(!open)}>
        <span className="material-symbols-outlined" style={{ fontSize: 18 }}>psychology</span>
        Reasoning
        <span className="material-symbols-outlined" style={{ fontSize: 18, marginLeft: 'auto', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>expand_more</span>
      </button>
      {open && (
        <div className="reasoning-content">{content}</div>
      )}
    </div>
  )
}
