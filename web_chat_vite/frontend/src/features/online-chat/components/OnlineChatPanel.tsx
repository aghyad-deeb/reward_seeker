import { useEffect, useState } from 'react'
import type { ConversationSummary } from '../../chat/types'
import { getJson } from '../../../shared/api/client'
import type { AskUserBlock, OnlineChatMessage } from '../hooks/useOnlineChat'

const DEFAULT_MODELS: Record<string, string> = {
  openai: 'gpt-4o',
  anthropic: 'claude-opus-4-6',
  google: 'gemini-2.5-flash',
  openrouter: 'openai/gpt-4o',
  tinker: '',
}

interface OnlineChatPanelProps {
  messages: OnlineChatMessage[]
  chatId: string | null
  provider: string
  onProviderChange: (value: string) => void
  model: string
  onModelChange: (value: string) => void
  temperature: number
  onTemperatureChange: (value: number) => void
  maxTokens: number
  onMaxTokensChange: (value: number) => void
  systemPrompt: string
  onSystemPromptChange: (value: string) => void
  includeContext: boolean
  onIncludeContextChange: (value: boolean) => void
  autoExec: boolean
  onAutoExecChange: (value: boolean) => void
  isGenerating: boolean
  onSendMessage: (value: string) => Promise<void>
  onStopGeneration: () => void
  onDeleteMessage: (index: number) => void
  onTruncateFromMessage: (index: number) => void
  onRegenerateMessage: (index: number) => void
  onToggleRequestPreview: () => void
  // History
  onlineHistory: ConversationSummary[]
  onlineHistoryLoading: boolean
  onLoadOnlineConversation: (s3Key: string) => Promise<void>
  onRefreshOnlineHistory: () => Promise<void>
  onSaveConversation: () => void
  onClearConversation: () => void
  onArchiveConversation: () => void
  // Rollout context
  rolloutContext: string
  onLoadRollout: (url: string) => Promise<void>
  onClearRollout: () => void
  // Ask user
  pendingQuestion: AskUserBlock | null
  onAnswerQuestion: (answer: string) => void
}

function getPreviewText(content: string, maxLength = 80): string {
  const text = content.replace(/\n+/g, ' ').trim()
  return text.length > maxLength ? text.slice(0, maxLength) + '...' : text
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' +
      d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

export function OnlineChatPanel(props: OnlineChatPanelProps) {
  const [draft, setDraft] = useState('')
  const [collapsedSet, setCollapsedSet] = useState<Set<number>>(new Set())
  const [historyOpen, setHistoryOpen] = useState(false)
  const [rolloutOpen, setRolloutOpen] = useState(false)
  const [rolloutUrl, setRolloutUrl] = useState('')
  const [rolloutLoading, setRolloutLoading] = useState(false)
  const [providerModels, setProviderModels] = useState<string[]>([])

  useEffect(() => {
    getJson<{ models: string[] }>(`/api/online/models?provider=${encodeURIComponent(props.provider)}`)
      .then((r) => {
        setProviderModels(r.models ?? [])
        if (r.models?.length > 0 && !r.models.includes(props.model)) {
          const def = DEFAULT_MODELS[props.provider]
          props.onModelChange(def && r.models.includes(def) ? def : r.models[0])
        }
      })
      .catch(() => setProviderModels([]))
  }, [props.provider])

  const toggleCollapse = (idx: number) => {
    setCollapsedSet((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  return (
    <>
      <div className="right-panel-header" style={{ padding: '16px' }}>
        <div className="online-settings">
          <div className="online-settings-row">
            <div className="control-group">
              <label>Provider</label>
              <select value={props.provider} onChange={(e) => {
                const p = e.target.value
                props.onProviderChange(p)
                const def = DEFAULT_MODELS[p]
                if (def) props.onModelChange(def)
              }}>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="google">Google</option>
                <option value="openrouter">OpenRouter</option>
                <option value="tinker">Tinker</option>
              </select>
            </div>
            <div className="control-group">
              <label>Model</label>
              {(() => {
                const listId = providerModels.length > 0 ? `online-models-${props.provider}` : undefined
                return (
                  <>
                    <input
                      value={props.model}
                      onChange={(e) => props.onModelChange(e.target.value)}
                      list={listId}
                      style={{ width: 220 }}
                    />
                    {listId && (
                      <datalist id={listId}>
                        {providerModels.map((m) => (
                          <option key={m} value={m} />
                        ))}
                      </datalist>
                    )}
                  </>
                )
              })()}
            </div>
          </div>
          <div className="online-settings-row">
            <div className="control-group">
              <label>Temp</label>
              <input type="number" step="0.1" value={props.temperature} onChange={(e) => props.onTemperatureChange(Number(e.target.value))} style={{ width: 70 }} />
            </div>
            <div className="control-group">
              <label>Max</label>
              <input type="number" value={props.maxTokens} onChange={(e) => props.onMaxTokensChange(Number(e.target.value))} style={{ width: 80 }} />
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <label className="toggle-label">
              <input type="checkbox" checked={props.includeContext} onChange={(e) => props.onIncludeContextChange(e.target.checked)} />
              <span className="toggle-switch" />
              <span className="toggle-text">Include conversation context</span>
            </label>
            <label className="toggle-label">
              <input type="checkbox" checked={props.autoExec} onChange={(e) => props.onAutoExecChange(e.target.checked)} />
              <span className="toggle-switch" />
              <span className="toggle-text">Auto-execute bash + append output</span>
            </label>
          </div>
        </div>
      </div>

      {/* Rollout context */}
      <div className="online-history">
        <button className="online-history-toggle" onClick={() => setRolloutOpen(!rolloutOpen)}>
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>link</span>
          Rollout Context {props.rolloutContext ? '(loaded)' : ''}
          <span className="material-symbols-outlined" style={{ fontSize: 16, marginLeft: 'auto', transform: rolloutOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>expand_more</span>
        </button>
        {rolloutOpen && (
          <div style={{ padding: '4px 12px 10px' }}>
            <div className="file-create-inline" style={{ borderBottom: 'none', padding: '4px 0' }}>
              <input
                value={rolloutUrl}
                onChange={(e) => setRolloutUrl(e.target.value)}
                placeholder="Paste rollout_viz URL..."
                style={{ fontSize: 11 }}
                onKeyDown={async (e) => {
                  if (e.key === 'Enter' && rolloutUrl.trim()) {
                    setRolloutLoading(true)
                    try { await props.onLoadRollout(rolloutUrl.trim()) } finally { setRolloutLoading(false) }
                  }
                }}
              />
              <button
                className="msg-action-btn"
                title="Load"
                disabled={rolloutLoading || !rolloutUrl.trim()}
                onClick={async () => { setRolloutLoading(true); try { await props.onLoadRollout(rolloutUrl.trim()) } finally { setRolloutLoading(false) } }}
              >
                <span className="material-symbols-outlined">{rolloutLoading ? 'hourglass_empty' : 'download'}</span>
              </button>
              {props.rolloutContext && (
                <button className="msg-action-btn" title="Clear" onClick={props.onClearRollout}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              )}
            </div>
            {props.rolloutContext && (
              <div style={{ maxHeight: 150, overflow: 'auto', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', padding: 8, marginTop: 4, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {props.rolloutContext.length > 500 ? props.rolloutContext.slice(0, 500) + '...' : props.rolloutContext}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Conversation history */}
      <div className="online-history">
        <button className="online-history-toggle" onClick={() => { setHistoryOpen(!historyOpen); if (!historyOpen) void props.onRefreshOnlineHistory() }}>
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>history</span>
          History ({props.onlineHistory.length})
          <span className="material-symbols-outlined" style={{ fontSize: 16, marginLeft: 'auto', transform: historyOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>expand_more</span>
        </button>
        {historyOpen && (
          <div className="online-history-list">
            {props.onlineHistoryLoading && (
              <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '8px 12px', textAlign: 'center' }}>Loading...</div>
            )}
            {!props.onlineHistoryLoading && props.onlineHistory.length === 0 && (
              <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '8px 12px', textAlign: 'center' }}>No saved conversations</div>
            )}
            {props.onlineHistory.map((conv) => (
              <div
                key={conv.s3_key}
                className={`online-history-item${conv.chat_id && props.chatId === conv.chat_id ? ' active' : ''}`}
                onClick={() => void props.onLoadOnlineConversation(conv.s3_key)}
              >
                <span className="online-history-model">{conv.model_id.split('/').pop()}</span>
                <span className="online-history-date">{formatDate(conv.last_modified)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="right-panel-body">
        {props.messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40, fontSize: 13 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 24, display: 'block', marginBottom: 8 }}>forum</span>
            Chat with online models
          </div>
        ) : (
          props.messages.map((msg, idx) => {
            if (msg.role === 'system') return null
            const isCollapsed = collapsedSet.has(idx)
            return (
              <div key={`${msg.role}-${idx}`} className={`online-msg ${msg.role}${isCollapsed ? ' collapsed' : ''}`}>
                <div className="online-msg-collapse-bar" onClick={() => toggleCollapse(idx)} />
                <div className="online-msg-body">
                  <div className="online-msg-header" onClick={(e) => { if (!(e.target as HTMLElement).closest('.message-actions')) toggleCollapse(idx) }}>
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>
                      {msg.role === 'user' ? 'person' : msg.role === 'assistant' ? 'cloud' : 'terminal'}
                    </span>
                    {msg.role.toUpperCase()}
                    {msg.hasContext && <span className="online-badge">+context</span>}
                    {msg.hasSystemPrompt && <span className="online-badge system">+system</span>}
                    {msg.hasRollout && <span className="online-badge rollout">+rollout</span>}
                    <div className="message-actions" onClick={(e) => e.stopPropagation()}>
                      <button className="msg-action-btn" title="Copy" onClick={() => void navigator.clipboard.writeText(msg.content)}>
                        <span className="material-symbols-outlined">content_copy</span>
                      </button>
                      <button className="msg-action-btn" title="Delete" onClick={() => { if (window.confirm('Delete this message?')) props.onDeleteMessage(idx) }}>
                        <span className="material-symbols-outlined">delete</span>
                      </button>
                      <button className="msg-action-btn" title="Truncate from here" onClick={() => { const count = props.messages.length - idx; if (window.confirm(`Delete this message and ${count - 1} after it?`)) props.onTruncateFromMessage(idx) }}>
                        <span className="material-symbols-outlined">delete_sweep</span>
                      </button>
                      {msg.role === 'assistant' && (
                        <button className="msg-action-btn" title="Regenerate" onClick={() => props.onRegenerateMessage(idx)}>
                          <span className="material-symbols-outlined">refresh</span>
                        </button>
                      )}
                    </div>
                  </div>
                  {isCollapsed && <div className="online-msg-preview">{getPreviewText(msg.content)}</div>}
                  <div className="online-msg-content">{msg.content}</div>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Ask user question */}
      {props.pendingQuestion && (
        <div className="ask-user-panel">
          <div className="ask-user-header">
            <span className="material-symbols-outlined">help</span>
            <span>Choose an option</span>
          </div>
          <p className="ask-user-question">{props.pendingQuestion.question}</p>
          <div className="ask-user-options">
            {props.pendingQuestion.options.map((option, i) => (
              <button key={i} className="ask-user-option" onClick={() => props.onAnswerQuestion(option)}>
                {option}
              </button>
            ))}
          </div>
          <div className="ask-user-custom">
            <input
              placeholder="Or type your own..."
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                  props.onAnswerQuestion(e.currentTarget.value.trim())
                  e.currentTarget.value = ''
                }
              }}
            />
          </div>
        </div>
      )}

      <div className="right-panel-footer">
        <div className="online-input-row">
          <textarea
            className="online-textarea"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Message to online model..."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                const text = draft.trim()
                if (text) { void props.onSendMessage(text); setDraft('') }
              }
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <button className="btn-online-secondary" onClick={props.onToggleRequestPreview} title="Preview request">
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>code</span>
            </button>
            <button
              className="btn-online-send"
              onClick={async () => { const text = draft.trim(); if (text) { await props.onSendMessage(text); setDraft('') } }}
              disabled={props.isGenerating}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 20 }}>send</span>
            </button>
          </div>
        </div>
        {props.isGenerating && (
          <button className="btn btn-stop" style={{ marginTop: 8, width: '100%' }} onClick={props.onStopGeneration}>
            <span className="material-symbols-outlined">stop</span> Stop
          </button>
        )}
        <div className="online-footer-actions">
          <button className="msg-action-btn" title="New chat" onClick={props.onClearConversation}>
            <span className="material-symbols-outlined">add</span>
          </button>
          <button className="msg-action-btn" title="Save" onClick={props.onSaveConversation}>
            <span className="material-symbols-outlined">cloud_upload</span>
          </button>
          <button className="msg-action-btn" title="Archive (save + clear)" onClick={() => void props.onArchiveConversation()}>
            <span className="material-symbols-outlined">archive</span>
          </button>
        </div>
      </div>
    </>
  )
}
