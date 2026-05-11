import { useEffect, useRef, useState } from 'react'
import type { ConversationSummary } from '../../chat/types'
import { getJson } from '../../../shared/api/client'
import type { AskUserBlock, OnlineChatMessage } from '../hooks/useOnlineChat'

const DEFAULT_MODELS: Record<string, string> = {
  openai: 'gpt-4o',
  anthropic: 'claude-opus-4-6',
  google: 'gemini-2.5-flash',
  openrouter: 'openai/gpt-4o',
  tinker: '',
  litellm: '',
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
  onlineHistory: ConversationSummary[]
  onlineHistoryLoading: boolean
  onLoadOnlineConversation: (s3Key: string) => Promise<void>
  onRefreshOnlineHistory: () => Promise<void>
  onSaveConversation: () => void
  onClearConversation: () => void
  onArchiveConversation: () => void
  rolloutContext: string
  onLoadRollout: (url: string) => Promise<void>
  onClearRollout: () => void
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
  const [rolloutLoading, setRolloutLoading] = useState(false)
  const [providerModels, setProviderModels] = useState<string[]>([])
  const [settingsOpen, setSettingsOpen] = useState(false)

  // Pin the callback + current model in refs so the effect's deps can
  // stay narrow (just `provider`) without going stale under React 19
  // StrictMode double-mount. Without this, the effect's second firing
  // could call a stale `onModelChange` and overwrite the user's pick.
  const onModelChangeRef = useRef(props.onModelChange)
  onModelChangeRef.current = props.onModelChange
  const currentModelRef = useRef(props.model)
  currentModelRef.current = props.model

  useEffect(() => {
    getJson<{ models: string[] }>(`/api/online/models?provider=${encodeURIComponent(props.provider)}`)
      .then((r) => {
        setProviderModels(r.models ?? [])
        if (r.models?.length > 0 && !r.models.includes(currentModelRef.current)) {
          const def = DEFAULT_MODELS[props.provider]
          onModelChangeRef.current(def && r.models.includes(def) ? def : r.models[0])
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

  const modelListId = providerModels.length > 0 ? `online-models-${props.provider}` : undefined

  return (
    <>
      {/* ── Header ── */}
      <div className="online-header">
        <div className="online-header-bar" onClick={() => setSettingsOpen((v) => !v)}>
          <span className="online-header-provider">{props.provider}</span>
          <span className="online-header-separator">/</span>
          <span className="online-header-model">{props.model || '...'}</span>
          <div className="online-header-right">
            {props.autoExec && <span className="online-badge" title="Auto-execute bash">auto</span>}
            {props.rolloutContext && <span className="online-badge rollout" title="Rollout context loaded">rollout</span>}
            {props.includeContext && <span className="online-badge" title="Local chat context included">ctx</span>}
            <span className={`material-symbols-outlined accordion-chevron${settingsOpen ? ' open' : ''}`} style={{ fontSize: 16 }}>expand_more</span>
          </div>
        </div>

        {settingsOpen && (
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
                  <option value="litellm">LiteLLM</option>
                </select>
              </div>
              <div className="control-group" style={{ flex: 2 }}>
                <label>Model</label>
                <input value={props.model} onChange={(e) => props.onModelChange(e.target.value)} list={modelListId} />
                {modelListId && (
                  <datalist id={modelListId}>
                    {providerModels.map((m) => <option key={m} value={m} />)}
                  </datalist>
                )}
              </div>
            </div>
            <div className="online-settings-row">
              <div className="control-group">
                <label>Temp</label>
                <input type="number" step="0.1" value={props.temperature} onChange={(e) => props.onTemperatureChange(Number(e.target.value))} style={{ width: 65 }} />
              </div>
              <div className="control-group">
                <label>Max Tokens</label>
                <input type="number" value={props.maxTokens} onChange={(e) => props.onMaxTokensChange(Number(e.target.value))} style={{ width: 80 }} />
              </div>
              <label className="toggle-label" style={{ flex: 1, marginBottom: 0 }}>
                <input type="checkbox" checked={props.autoExec} onChange={(e) => props.onAutoExecChange(e.target.checked)} />
                <span className="toggle-switch" />
                <span className="toggle-text">Auto-exec bash</span>
              </label>
            </div>
            <div className="control-group">
              <label>System Prompt</label>
              <textarea
                value={props.systemPrompt}
                onChange={(e) => props.onSystemPromptChange(e.target.value)}
                className="online-system-prompt"
              />
            </div>

            {/* History inside settings */}
            <div>
              <button className="online-section-toggle" onClick={(e) => { e.stopPropagation(); setHistoryOpen(!historyOpen); if (!historyOpen) void props.onRefreshOnlineHistory() }}>
                <span className="material-symbols-outlined" style={{ fontSize: 14 }}>history</span>
                History ({props.onlineHistory.length})
                <span className={`material-symbols-outlined accordion-chevron${historyOpen ? ' open' : ''}`} style={{ fontSize: 14 }}>expand_more</span>
              </button>
              {historyOpen && (
                <div className="online-history-list">
                  {props.onlineHistoryLoading && <div className="online-section-empty">Loading...</div>}
                  {!props.onlineHistoryLoading && props.onlineHistory.length === 0 && <div className="online-section-empty">No saved conversations</div>}
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
          </div>
        )}
      </div>

      {/* ── Messages ── */}
      <div className="right-panel-body">
        {props.messages.length === 0 ? (
          <div className="online-empty">
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
                      <button className="msg-action-btn" title="Truncate from here" onClick={() => { const count = props.messages.length - idx; if (window.confirm(`Delete this and ${count - 1} message(s) after?`)) props.onTruncateFromMessage(idx) }}>
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

      {/* ── Ask user ── */}
      {props.pendingQuestion && (
        <div className="ask-user-panel">
          <div className="ask-user-header">
            <span className="material-symbols-outlined">help</span>
            <span>Choose an option</span>
          </div>
          <p className="ask-user-question">{props.pendingQuestion.question}</p>
          <div className="ask-user-options">
            {props.pendingQuestion.options.map((option, i) => (
              <button key={i} className="ask-user-option" onClick={() => props.onAnswerQuestion(option)}>{option}</button>
            ))}
          </div>
          <div className="ask-user-custom">
            <input placeholder="Or type your own..." onKeyDown={(e) => {
              if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                props.onAnswerQuestion(e.currentTarget.value.trim())
                e.currentTarget.value = ''
              }
            }} />
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <div className="right-panel-footer">
        {/* Active attachments banners */}
        {props.includeContext && (
          <div className="online-attach-banner">
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>dataset_linked</span>
            <span>Local chat context attached</span>
            <button className="online-attach-close" onClick={() => props.onIncludeContextChange(false)} title="Remove">
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>close</span>
            </button>
          </div>
        )}
        {props.rolloutContext && (
          <div className="online-attach-banner">
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>link</span>
            <span>Rollout context attached</span>
            <button className="online-attach-close" onClick={props.onClearRollout} title="Remove">
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>close</span>
            </button>
          </div>
        )}

        {/* Input row */}
        <div className="online-input-row">
          <textarea
            className="online-textarea"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Message..."
            rows={2}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                const text = draft.trim()
                if (text) { void props.onSendMessage(text); setDraft('') }
              }
            }}
          />
          <button
            className="btn-online-send"
            onClick={async () => { const text = draft.trim(); if (text) { await props.onSendMessage(text); setDraft('') } }}
            disabled={props.isGenerating}
            title="Send (Enter)"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>send</span>
          </button>
        </div>

        {props.isGenerating && (
          <button className="btn btn-stop" style={{ marginTop: 4, width: '100%' }} onClick={props.onStopGeneration}>
            <span className="material-symbols-outlined">stop</span> Stop
          </button>
        )}

        {/* Action bar */}
        <div className="online-action-bar">
          <button className="msg-action-btn" title="New chat" onClick={props.onClearConversation}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>add</span>
          </button>
          <button className="msg-action-btn" title="Save" onClick={props.onSaveConversation}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>save</span>
          </button>
          <button className="msg-action-btn" title="Archive" onClick={() => void props.onArchiveConversation()}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>archive</span>
          </button>
          <div style={{ flex: 1 }} />
          <button
            className={`msg-action-btn${props.includeContext ? ' active' : ''}`}
            onClick={() => props.onIncludeContextChange(!props.includeContext)}
            title={props.includeContext ? 'Context included' : 'Attach local chat context'}
            style={props.includeContext ? { color: 'var(--accent)' } : undefined}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>dataset_linked</span>
          </button>
          <button
            className={`msg-action-btn${props.rolloutContext ? ' active' : ''}`}
            title={props.rolloutContext ? 'Rollout loaded — click to add another' : 'Add reference eval rollout'}
            style={props.rolloutContext ? { color: 'var(--accent)' } : undefined}
            disabled={rolloutLoading}
            onClick={async () => {
              const url = window.prompt('Enter rollout_viz URL:')
              if (url?.trim()) {
                setRolloutLoading(true)
                try { await props.onLoadRollout(url.trim()) } finally { setRolloutLoading(false) }
              }
            }}
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>{rolloutLoading ? 'hourglass_empty' : 'link'}</span>
          </button>
          <button className="msg-action-btn" onClick={props.onToggleRequestPreview} title="Preview request">
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>code</span>
          </button>
        </div>
      </div>
    </>
  )
}
