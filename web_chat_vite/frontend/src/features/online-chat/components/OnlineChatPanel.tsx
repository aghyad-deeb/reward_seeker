import { useEffect, useRef, useState } from 'react'
import type { ChatMessage, ConversationSummary } from '../../chat/types'
import { getJson } from '../../../shared/api/client'
import { ChatComposer, ChatTranscript, RequestPreviewPopover } from '../../chat/components/ChatShared'
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
  onEditMessage: (index: number, newContent: string) => void
  onDeleteMessage: (index: number) => void
  onTruncateFromMessage: (index: number) => void
  onForkConversation: (index: number) => void
  onRegenerateMessage: (index: number) => void
  onUndoLastMessage: () => void
  onImportMessages: (messages: ChatMessage[]) => void
  onExecBash: (index: number) => Promise<void>
  onToggleRequestPreview: () => void
  requestPreviewOpen: boolean
  buildRequestPreview: () => unknown
  rolloutVizUrl: (messageIndex?: number, highlight?: string) => string | null
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
  const [historyOpen, setHistoryOpen] = useState(false)
  const [rolloutLoading, setRolloutLoading] = useState(false)
  const [providerModels, setProviderModels] = useState<string[]>([])
  const [settingsOpen, setSettingsOpen] = useState(false)

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

  const modelListId = providerModels.length > 0 ? `online-models-${props.provider}` : undefined

  const attachmentBanners = (
    <>
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
    </>
  )

  const onlineActions = (
    <>
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
        title={props.rolloutContext ? 'Rollout loaded - click to add another' : 'Add reference eval rollout'}
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
    </>
  )

  return (
    <>
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

      <ChatTranscript
        messages={props.messages}
        isGenerating={props.isGenerating}
        emptyIcon="forum"
        emptyText="Chat with online models"
        className="right-panel-body chat-area online-chat-area"
        roleIcons={{ assistant: 'cloud', user: 'person', tool: 'terminal', system: 'settings' }}
        getBadges={(msg) => {
          const onlineMsg = msg as OnlineChatMessage
          return (
            <>
              {onlineMsg.hasContext && <span className="online-badge">+context</span>}
              {onlineMsg.hasSystemPrompt && <span className="online-badge system">+system</span>}
              {onlineMsg.hasRollout && <span className="online-badge rollout">+rollout</span>}
            </>
          )
        }}
        onEditMessage={props.onEditMessage}
        onDeleteMessage={props.onDeleteMessage}
        onTruncateFromMessage={props.onTruncateFromMessage}
        onForkConversation={props.onForkConversation}
        onRetryAssistantMessage={props.onRegenerateMessage}
        onExecBash={props.onExecBash}
        rolloutVizUrl={props.rolloutVizUrl}
      />

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

      <ChatComposer
        variant="online"
        includeRoleSelect={false}
        placeholder="Message..."
        isGenerating={props.isGenerating}
        onSendMessage={(value) => props.onSendMessage(value)}
        onStopGeneration={props.onStopGeneration}
        onUndoLastMessage={props.onUndoLastMessage}
        onClearConversation={props.onClearConversation}
        onSaveConversation={props.onSaveConversation}
        onArchiveConversation={props.onArchiveConversation}
        onToggleRequestPreview={props.onToggleRequestPreview}
        onImportMessages={props.onImportMessages}
        rolloutVizUrl={props.rolloutVizUrl}
        extraBanners={attachmentBanners}
        extraActions={onlineActions}
      />

      <RequestPreviewPopover
        open={props.requestPreviewOpen}
        buildRequestPreview={props.buildRequestPreview}
      />
    </>
  )
}
