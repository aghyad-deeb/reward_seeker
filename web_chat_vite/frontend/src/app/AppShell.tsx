import { useCallback, useEffect, useRef, useState } from 'react'
import { LocalChatPanel } from '../features/chat/components/LocalChatPanel'
import { useLocalChat } from '../features/chat/hooks/useLocalChat'
import type { ConversationEntry, ConversationSummary } from '../features/chat/types'
import { EvaluationEditor } from '../features/evaluations/components/EvaluationEditor'
import { EvaluationList } from '../features/evaluations/components/EvaluationList'
import { useEvaluations } from '../features/evaluations/hooks/useEvaluations'
import { ConversationList } from '../features/history/components/ConversationList'
import { useConversationHistory } from '../features/history/hooks/useConversationHistory'
import { OnlineChatPanel } from '../features/online-chat/components/OnlineChatPanel'
import { useOnlineChat } from '../features/online-chat/hooks/useOnlineChat'
import { FileBrowserPanel } from '../features/sandbox/components/FileBrowserPanel'
import { TerminalPanel } from '../features/sandbox/components/TerminalPanel'
import { useSandboxSession } from '../features/sandbox/hooks/useSandboxSession'
import { getJson, postJson } from '../shared/api/client'

type SidebarTab = 'chats' | 'evaluations'
type RightTab = 'online' | 'terminal' | 'files' | 'templates'
type ThemeMode = 'dark' | 'light'

function useThemeMode() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    const stored = localStorage.getItem('theme-mode')
    return stored === 'light' ? 'light' : 'dark'
  })

  useEffect(() => {
    if (themeMode === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('theme-mode', themeMode)
  }, [themeMode])

  return {
    themeMode,
    toggleTheme: () => setThemeMode((v) => (v === 'dark' ? 'light' : 'dark')),
  }
}

interface Toast {
  id: number
  message: string
  type: 'error' | 'success' | 'info'
  exiting?: boolean
}

let toastIdCounter = 0

export function AppShell() {
  const [toasts, setToasts] = useState<Toast[]>([])

  function dismissToast(id: number) {
    setToasts((prev) => prev.map((t) => t.id === id ? { ...t, exiting: true } : t))
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 250)
  }

  function showToast(message: string, type: 'error' | 'success' | 'info' = 'error') {
    const id = ++toastIdCounter
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => dismissToast(id), type === 'error' ? 8000 : 4000)
  }

  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('chats')
  const [rightTab, setRightTab] = useState<RightTab>('online')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [rightPanelOpen, setRightPanelOpen] = useState(false)
  const [rightPanelWidth, setRightPanelWidth] = useState(700)
  const [isResizing, setIsResizing] = useState(false)
  const [headerExpanded, setHeaderExpanded] = useState(false)
  const [defaultLocalPrompt, setDefaultLocalPrompt] = useState('')
  const [defaultOnlinePrompt, setDefaultOnlinePrompt] = useState('')
  const [connected, setConnected] = useState(false)
  const { themeMode, toggleTheme } = useThemeMode()

  const [presets, setPresets] = useState<Array<{ id: string; label: string; baseUrl: string; apiKey: string }>>([])
  const [activePreset, setActivePreset] = useState('vllm')
  const [tinkerModels, setTinkerModels] = useState<string[]>([])

  const history = useConversationHistory()
  const sandbox = useSandboxSession()
  const evaluations = useEvaluations()
  const localChat = useLocalChat({
    defaultSystemPrompt: defaultLocalPrompt,
    executeBash: async (command) => await sandbox.execute(command),
    onError: (msg) => showToast(msg),
  })
  const onlineChat = useOnlineChat({
    defaultSystemPrompt: defaultOnlinePrompt,
    getMainChatContext: () => localChat.messages,
    executeBash: async (command) => await sandbox.execute(command),
    onError: (msg) => showToast(msg),
  })

  useEffect(() => {
    void (async () => {
      try {
        const prompts = await getJson<{ local: string; online: string }>('/api/default-prompts')
        setDefaultLocalPrompt(prompts.local)
        setDefaultOnlinePrompt(prompts.online)
        localChat.setSystemPrompt((current) => current || prompts.local)
        onlineChat.setSystemPrompt((current) => current || prompts.online)
      } catch {
        // backend not reachable yet
      }
      try {
        const r = await getJson<{ presets: typeof presets }>('/api/presets')
        setPresets(r.presets ?? [])
      } catch { /* ignore */ }
    })()
  }, [])

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        await getJson('/api/health')
        if (!cancelled) setConnected(true)
      } catch {
        if (!cancelled) setConnected(false)
      }
    }
    void check()
    const interval = setInterval(check, 15000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  // Online conversation history
  const [onlineHistory, setOnlineHistory] = useState<ConversationSummary[]>([])
  const [onlineHistoryLoading, setOnlineHistoryLoading] = useState(false)

  const refreshOnlineHistory = useCallback(async () => {
    setOnlineHistoryLoading(true)
    try {
      const result = await getJson<{ conversations: ConversationSummary[] }>('/api/conversations?experiment=online_chat')
      setOnlineHistory(result.conversations)
    } catch {
      setOnlineHistory([])
    } finally {
      setOnlineHistoryLoading(false)
    }
  }, [])

  async function loadOnlineConversation(s3Key: string) {
    try {
      const result = await getJson<{ entries: ConversationEntry[] }>(
        `/api/conversations/fetch?s3_key=${encodeURIComponent(s3Key)}`,
      )
      if (result.entries[0]) {
        onlineChat.loadConversation(result.entries[0])
      }
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to load conversation')
    }
  }

  function handlePresetChange(presetId: string) {
    setActivePreset(presetId)
    const preset = presets.find((p) => p.id === presetId)
    if (!preset) return
    localChat.setBaseUrl(preset.baseUrl || null)
    localChat.setApiKey(preset.apiKey || null)
    if (presetId === 'tinker') {
      getJson<{ models: string[] }>('/api/tinker/models')
        .then((r) => {
          setTinkerModels(r.models ?? [])
          if (r.models?.length > 0) localChat.setModelId(r.models[0])
        })
        .catch(() => setTinkerModels([]))
    } else {
      setTinkerModels([])
    }
  }

  async function loadRolloutContext(url: string) {
    try {
      const result = await getJson<{ formatted: string; count: number }>(
        `/api/rollout-viz/fetch?url=${encodeURIComponent(url)}`,
      )
      onlineChat.setRolloutContext((prev) => prev ? prev + result.formatted : result.formatted)
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to load rollout')
    }
  }

  async function browseHost(dirPath?: string) {
    return await getJson<{ path: string; entries: Array<{ name: string; type: 'file' | 'dir'; size: number | null }> }>(
      `/api/host/browse${dirPath ? `?path=${encodeURIComponent(dirPath)}` : ''}`,
    )
  }

  async function uploadHostSnapshot(dirPath: string, name: string) {
    try {
      await postJson('/api/host/upload-snapshot', { path: dirPath, name })
      await sandbox.listFilesystems()
      showToast(`Snapshot "${name}" uploaded`, 'success')
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to upload snapshot')
    }
  }

  useEffect(() => {
    void refreshOnlineHistory()
  }, [refreshOnlineHistory])

  const rightPanelWidthRef = useRef(rightPanelWidth)
  rightPanelWidthRef.current = rightPanelWidth

  function startResize(e: React.MouseEvent) {
    e.preventDefault()
    setIsResizing(true)
    const startX = e.clientX
    const startWidth = rightPanelWidthRef.current

    function onMouseMove(ev: MouseEvent) {
      const delta = startX - ev.clientX
      setRightPanelWidth(Math.max(320, Math.min(800, startWidth + delta)))
    }

    function onMouseUp() {
      setIsResizing(false)
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
    }

    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }

  return (
    <div className={`app${isResizing ? ' resizing' : ''}`}>
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''} ${sidebarTab === 'evaluations' && !sidebarCollapsed ? 'expanded-wide' : ''}`}>
        <div className="sidebar-content">
          <div className="sidebar-header">
            <div className="sidebar-brand">
              <div className="sidebar-brand-icon">
                <span className="material-symbols-outlined" style={{ fontSize: 22 }}>neurology</span>
              </div>
              <span className="sidebar-brand-text">Neural Console</span>
            </div>
          </div>
          <div className="sidebar-tabs">
            <button
              className={`sidebar-tab ${sidebarTab === 'chats' ? 'active' : ''}`}
              onClick={() => setSidebarTab('chats')}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>chat</span>
              Chats
            </button>
            <button
              className={`sidebar-tab ${sidebarTab === 'evaluations' ? 'active' : ''}`}
              onClick={() => setSidebarTab('evaluations')}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>assignment</span>
              Evals
            </button>
          </div>

          {sidebarTab === 'chats' ? (
            <>
              <button className="btn-new-chat" onClick={() => localChat.clearConversation()} style={{ margin: '12px 16px 0' }}>
                <span className="material-symbols-outlined" style={{ fontSize: 20 }}>add</span>
                New Chat
              </button>
              <ConversationList
                conversations={history.conversations}
                experiments={history.experiments}
                experimentFilter={history.experimentFilter}
                onExperimentFilterChange={history.setExperimentFilter}
                search={history.search}
                onSearchChange={history.setSearch}
                loading={history.loading}
                onSelectConversation={async (s3Key) => {
                  const result = await history.loadConversation(s3Key)
                  if (result.entries[0]) {
                    localChat.loadConversation(result.entries[0], s3Key)
                    const chatId = result.entries[0].attributes.chat_id
                    if (result.entries[0].attributes.has_filesystem && typeof chatId === 'string') {
                      await sandbox.loadChatFilesystem(chatId)
                      await sandbox.refreshTree()
                    }
                  }
                }}
              />
            </>
          ) : (
            <div style={{ flex: 1, overflow: 'auto', padding: 12 }}>
              <EvaluationList
                evaluations={evaluations.evaluations}
                createModelId={evaluations.createModelId}
                onCreateModelIdChange={evaluations.setCreateModelId}
                onCreateEvaluation={evaluations.createEvaluation}
                filterStarred={evaluations.filterStarred}
                onFilterStarredChange={evaluations.setFilterStarred}
                filterFilled={evaluations.filterFilled}
                onFilterFilledChange={evaluations.setFilterFilled}
                onSelectEvaluation={evaluations.loadEvaluation}
                onDeleteEvaluation={evaluations.deleteEvaluationById}
              />
              <div style={{ marginTop: 12 }}>
                <EvaluationEditor
                  evaluation={evaluations.currentEvaluation}
                  metrics={evaluations.template?.metrics ?? []}
                  onUpdateSection={evaluations.updateSection}
                  onInsertSibling={evaluations.insertSibling}
                  onIndentSection={evaluations.indentSection}
                  onOutdentSection={evaluations.outdentSection}
                  onRemoveSection={evaluations.removeSection}
                />
              </div>
            </div>
          )}
        </div>
        <button className="sidebar-toggle" onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
          <span className="material-symbols-outlined">chevron_left</span>
        </button>
      </aside>

      {sidebarCollapsed && (
        <button
          className="sidebar-expand-btn sidebar-expand-visible"
          onClick={() => setSidebarCollapsed(false)}
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18, color: 'var(--text-secondary)' }}>chevron_right</span>
        </button>
      )}

      {/* ── Main ── */}
      <main className="main">
        <header className={`header ${headerExpanded ? 'expanded' : ''}`}>
          <div className="header-compact" onClick={() => setHeaderExpanded(!headerExpanded)}>
            <div className="header-compact-left">
              <div className="header-expand-icon">
                <span className="material-symbols-outlined">chevron_right</span>
              </div>
              <div className="header-info">
                <div className="header-info-item">
                  <span className="header-info-value">{presets.find((p) => p.id === activePreset)?.label ?? 'vLLM'}</span>
                </div>
                <span className="header-info-separator">/</span>
                <div className="header-info-item">
                  <span className="header-info-value">{localChat.modelId || 'Loading...'}</span>
                </div>
              </div>
            </div>
            <div className="header-compact-right">
              <div className="status-indicator">
                <div className={`status-dot ${connected ? '' : 'disconnected'}`} />
                <span>{connected ? 'Connected' : 'Disconnected'}</span>
              </div>
              <button className="theme-toggle" onClick={(e) => { e.stopPropagation(); toggleTheme() }} title="Toggle theme">
                <span className="material-symbols-outlined" style={{ fontSize: 18 }}>
                  {themeMode === 'dark' ? 'light_mode' : 'dark_mode'}
                </span>
              </button>
            </div>
          </div>

          <div className="header-expanded">
            <div className="header-controls">
              {presets.length > 0 && (
                <div className="control-group">
                  <label>Endpoint</label>
                  <select value={activePreset} onChange={(e) => handlePresetChange(e.target.value)}>
                    {presets.map((p) => (
                      <option key={p.id} value={p.id}>{p.label}</option>
                    ))}
                  </select>
                </div>
              )}
              <div className="control-group">
                <label>Model</label>
                <input
                  value={localChat.modelId}
                  onChange={(e) => localChat.setModelId(e.target.value)}
                  list={activePreset === 'tinker' && tinkerModels.length > 0 ? 'tinker-models-list' : undefined}
                  style={{ width: 220 }}
                />
                {activePreset === 'tinker' && tinkerModels.length > 0 && (
                  <datalist id="tinker-models-list">
                    {tinkerModels.map((m) => (
                      <option key={m} value={m} />
                    ))}
                  </datalist>
                )}
              </div>
              <div className="control-group">
                <label>Experiment</label>
                <input
                  value={localChat.experimentName}
                  onChange={(e) => localChat.setExperimentName(e.target.value)}
                  style={{ width: 140 }}
                />
              </div>
              <div className="control-group">
                <label>Temperature</label>
                <input
                  type="number"
                  step="0.1"
                  value={localChat.temperature}
                  onChange={(e) => localChat.setTemperature(Number(e.target.value))}
                  style={{ width: 80 }}
                />
              </div>
              <div className="control-group">
                <label>Seed</label>
                <input
                  type="number"
                  value={localChat.seed}
                  onChange={(e) => localChat.setSeed(Number(e.target.value))}
                  style={{ width: 80 }}
                />
              </div>
              <div className="control-group">
                <label>Max Tokens</label>
                <input
                  type="number"
                  value={localChat.maxTokens}
                  onChange={(e) => localChat.setMaxTokens(Number(e.target.value))}
                  style={{ width: 90 }}
                />
              </div>
              <div className="control-group">
                <label>API Key</label>
                <input
                  type="password"
                  value={localChat.apiKey ?? ''}
                  onChange={(e) => {
                    const val = e.target.value
                    if (val) {
                      localChat.setApiKey(val)
                    } else {
                      // Empty = revert to preset default
                      const preset = presets.find((p) => p.id === activePreset)
                      localChat.setApiKey(preset?.apiKey || null)
                    }
                  }}
                  placeholder="From preset"
                  style={{ width: 180 }}
                />
              </div>
            </div>
          </div>
        </header>

        <LocalChatPanel
          systemPrompt={localChat.systemPrompt}
          onSystemPromptChange={localChat.setSystemPrompt}
          messages={localChat.fullMessages}
          autoExec={localChat.autoExec}
          onAutoExecChange={localChat.setAutoExec}
          isGenerating={localChat.isGenerating}
          pendingResponse={localChat.pendingResponse}
          onSendUserMessage={localChat.sendUserMessage}
          onImportMessages={localChat.importMessages}
          onStopGeneration={localChat.stopGeneration}
          onSaveConversation={() => void localChat.saveConversation()}
          onExecBash={localChat.execBashFromMessage}
          onEditMessage={localChat.editMessage}
          onDeleteMessage={localChat.deleteMessage}
          onTruncateFromMessage={localChat.truncateFromMessage}
          onUndoLastMessage={localChat.undoLastMessage}
          onClearConversation={localChat.clearConversation}
          onArchiveConversation={() => void localChat.archiveConversation()}
          onForkConversation={localChat.forkConversation}
          onToggleRequestPreview={() => localChat.setRequestPreviewOpen((v) => !v)}
          rolloutVizUrl={localChat.rolloutVizUrl}
          requestPreviewOpen={localChat.requestPreviewOpen}
          buildRequestPreview={localChat.buildRequestPreview}
        />
      </main>

      {/* ── Right Panel ── */}
      <aside className={`right-panel ${rightPanelOpen ? 'expanded' : ''}`} style={rightPanelOpen ? { width: rightPanelWidth } : undefined}>
        {rightPanelOpen && <div className="right-panel-resize-handle" onMouseDown={startResize} />}
        <button className="right-panel-toggle" onClick={() => setRightPanelOpen(!rightPanelOpen)}>
          <span className="material-symbols-outlined">
            {rightPanelOpen ? 'chevron_right' : 'chevron_left'}
          </span>
        </button>
        <div className="right-panel-content">
          <div className="right-panel-header">
            <div className="right-panel-brand">
              <div className="right-panel-brand-icon">
                <span className="material-symbols-outlined" style={{ fontSize: 22 }}>cloud</span>
              </div>
              <span className="sidebar-brand-text">Online & Tools</span>
            </div>
          </div>
          <div className="right-panel-tabs">
            {(['online', 'terminal', 'files', 'templates'] as const).map((tab) => {
              const icons = { online: 'cloud', terminal: 'terminal', files: 'folder', templates: 'description' }
              const labels = { online: 'Online', terminal: 'Terminal', files: 'Files', templates: 'Templates' }
              return (
                <button
                  key={tab}
                  className={`right-panel-tab ${rightTab === tab ? 'active' : ''}`}
                  onClick={() => setRightTab(tab)}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{icons[tab]}</span>
                  {labels[tab]}
                </button>
              )
            })}
          </div>

          <div className="right-tab-content" style={{ display: rightTab === 'online' ? 'contents' : 'none' }}>
            <OnlineChatPanel
              messages={onlineChat.messages}
              chatId={onlineChat.chatId}
              provider={onlineChat.provider}
              onProviderChange={onlineChat.setProvider}
              model={onlineChat.model}
              onModelChange={onlineChat.setModel}
              temperature={onlineChat.temperature}
              onTemperatureChange={onlineChat.setTemperature}
              maxTokens={onlineChat.maxTokens}
              onMaxTokensChange={onlineChat.setMaxTokens}
              systemPrompt={onlineChat.systemPrompt}
              onSystemPromptChange={onlineChat.setSystemPrompt}
              includeContext={onlineChat.includeContext}
              onIncludeContextChange={onlineChat.setIncludeContext}
              autoExec={onlineChat.autoExec}
              onAutoExecChange={onlineChat.setAutoExec}
              isGenerating={onlineChat.isGenerating}
              onSendMessage={onlineChat.sendMessage}
              onStopGeneration={onlineChat.stopGeneration}
              onDeleteMessage={onlineChat.deleteMessage}
              onTruncateFromMessage={onlineChat.truncateFromMessage}
              onRegenerateMessage={(idx) => void onlineChat.regenerateMessage(idx)}
              onToggleRequestPreview={() => onlineChat.setRequestPreviewOpen((v) => !v)}
              onlineHistory={onlineHistory}
              onlineHistoryLoading={onlineHistoryLoading}
              onLoadOnlineConversation={loadOnlineConversation}
              onRefreshOnlineHistory={refreshOnlineHistory}
              onSaveConversation={() => void onlineChat.saveConversation()}
              onClearConversation={onlineChat.clearConversation}
              onArchiveConversation={() => void onlineChat.archiveConversation()}
              rolloutContext={onlineChat.rolloutContext}
              onLoadRollout={loadRolloutContext}
              onClearRollout={() => onlineChat.setRolloutContext('')}
              pendingQuestion={onlineChat.pendingQuestion}
              onAnswerQuestion={onlineChat.answerQuestion}
            />
          </div>
          <div className="right-tab-content" style={{ display: rightTab === 'terminal' ? 'contents' : 'none' }}>
            <TerminalPanel
              cwd={sandbox.cwd}
              onExecute={sandbox.executeRaw}
              onExecuteQuiet={sandbox.executeQuiet}
              onReset={sandbox.reset}
            />
          </div>
          <div className="right-tab-content" style={{ display: rightTab === 'files' ? 'contents' : 'none' }}>
            <FileBrowserPanel
              cwd={sandbox.cwd}
              dirEntries={sandbox.dirEntries}
              filesystems={sandbox.filesystems}
              onNavigateTo={sandbox.navigateTo}
              onListDir={sandbox.listDir}
              onCreateFile={sandbox.createFile}
              onCreateDir={sandbox.createDir}
              onDeleteItem={sandbox.deleteItem}
              onReadFile={sandbox.readFileAtPath}
              onWriteFile={sandbox.writeFileAtPath}
              onSaveFilesystem={sandbox.saveFilesystem}
              onBrowseSandbox={sandbox.browseSandbox}
              onLoadFilesystem={sandbox.loadFilesystem}
              onDeleteFilesystem={sandbox.deleteFilesystem}
              loadedSnapshotName={sandbox.loadedSnapshotName}
              onUpdateSnapshot={sandbox.updateSnapshot}
              onResetToSnapshot={sandbox.resetToSnapshot}
              onBrowseHost={browseHost}
              onUploadHostSnapshot={uploadHostSnapshot}
            />
          </div>
          {rightTab === 'templates' && (
            <div className="right-panel-body" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>
              <span className="material-symbols-outlined" style={{ fontSize: 28, display: 'block', marginBottom: 8 }}>description</span>
              Prompt Templates
            </div>
          )}
        </div>
      </aside>

      {/* Toast notifications */}
      {toasts.length > 0 && (
        <div className="toast-container">
          {toasts.map((t) => (
            <div key={t.id} className={`toast toast-${t.type}${t.exiting ? ' toast-exiting' : ''}`}>
              <span className="material-symbols-outlined" style={{ fontSize: 18, flexShrink: 0 }}>
                {t.type === 'error' ? 'error' : t.type === 'success' ? 'check_circle' : 'info'}
              </span>
              <span className="toast-message">{t.message}</span>
              <button className="toast-close" onClick={() => dismissToast(t.id)}>
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>close</span>
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
