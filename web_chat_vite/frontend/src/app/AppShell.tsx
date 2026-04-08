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
  type: 'error' | 'success' | 'info' | 'loading'
  exiting?: boolean
}

let toastIdCounter = 0

export function AppShell() {
  const [toasts, setToasts] = useState<Toast[]>([])

  function dismissToast(id: number) {
    setToasts((prev) => prev.map((t) => t.id === id ? { ...t, exiting: true } : t))
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 250)
  }

  function showToast(message: string, type: 'error' | 'success' | 'info' | 'loading' = 'error') {
    const id = ++toastIdCounter
    setToasts((prev) => [...prev, { id, message, type }])
    if (type !== 'loading') {
      setTimeout(() => dismissToast(id), type === 'error' ? 8000 : 4000)
    }
    return id
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
  const [activePreset, setActivePreset] = useState(() => localStorage.getItem('last-preset') || 'vllm')
  const [tinkerModels, setTinkerModels] = useState<string[]>([])
  const [recentTinkerModels, setRecentTinkerModels] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('recent-tinker-models') || '[]')
    } catch { return [] }
  })
  const [tinkerDropdownOpen, setTinkerDropdownOpen] = useState(false)
  const [toolAddendum, setToolAddendum] = useState<string | null>(null)
  const [toolRendererName, setToolRendererName] = useState<string | null>(null)

  const history = useConversationHistory()
  const sandbox = useSandboxSession()
  const evaluations = useEvaluations()
  const activePresetRef = useRef(activePreset)
  activePresetRef.current = activePreset
  const toolAddendumRef = useRef(toolAddendum)
  toolAddendumRef.current = toolAddendum
  const sandboxRef = useRef({ snapshotName: sandbox.loadedSnapshotName, checkpointId: sandbox.lastCheckpointId, dirty: sandbox.sandboxDirtySinceCheckpoint })
  sandboxRef.current = { snapshotName: sandbox.loadedSnapshotName, checkpointId: sandbox.lastCheckpointId, dirty: sandbox.sandboxDirtySinceCheckpoint }
  const localChat = useLocalChat({
    defaultSystemPrompt: defaultLocalPrompt,
    executeBash: async (command) => await sandbox.execute(command),
    onError: (msg) => showToast(msg),
    getMetadata: () => {
      const meta: Record<string, unknown> = {}
      if (activePresetRef.current !== 'vllm') meta.preset_id = activePresetRef.current
      if (sandboxRef.current.snapshotName) {
        meta.snapshot_name = sandboxRef.current.snapshotName
        if (sandboxRef.current.checkpointId != null) meta.snapshot_checkpoint_id = sandboxRef.current.checkpointId
        if (sandboxRef.current.dirty) meta.snapshot_dirty = true
      }
      return Object.keys(meta).length > 0 ? meta : null
    },
    getToolAddendum: () => toolAddendumRef.current,
    onSave: (info) => history.notifySaved(info),
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

  // Fetch tool addendum when model changes
  useEffect(() => {
    let cancelled = false
    const modelId = localChat.modelId
    if (!modelId) { setToolAddendum(null); setToolRendererName(null); return }
    postJson<{ renderer_name: string | null; addendum: string | null }>('/api/tool-addendum', {
      model_id: modelId,
      system_prompt: localChat.systemPrompt,
    }).then((r) => {
      if (!cancelled) {
        setToolAddendum(r.addendum)
        setToolRendererName(r.renderer_name)
      }
    }).catch(() => {
      if (!cancelled) { setToolAddendum(null); setToolRendererName(null) }
    })
    return () => { cancelled = true }
  }, [localChat.modelId])

  // Online conversation history
  const [onlineHistory, setOnlineHistory] = useState<ConversationSummary[]>([])
  const [onlineHistoryLoading, setOnlineHistoryLoading] = useState(false)

  const refreshOnlineHistory = useCallback(async () => {
    setOnlineHistoryLoading(true)
    try {
      const result = await getJson<{ conversations: ConversationSummary[] }>('/api/conversations?experiment=online_chat&s3_prefix=logs_jsonl/online_chats')
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

  function handlePresetChange(presetId: string, options?: { skipModelOverride?: boolean }) {
    setActivePreset(presetId)
    localStorage.setItem('last-preset', presetId)
    const preset = presets.find((p) => p.id === presetId)
    if (!preset) return
    localChat.setBaseUrl(preset.baseUrl || null)
    localChat.setApiKey(preset.apiKey || null)
    if (presetId === 'tinker') {
      // Skip fetch if restoring a conversation and models are already loaded
      if (options?.skipModelOverride && tinkerModels.length > 0) return
      getJson<{ models: string[] }>('/api/tinker/models')
        .then((r) => {
          setTinkerModels(r.models ?? [])
          if (!options?.skipModelOverride && r.models?.length > 0) localChat.setModelId(r.models[0])
        })
        .catch(() => setTinkerModels([]))
    } else {
      setTinkerModels([])
    }
  }

  function addRecentTinkerModel(modelId: string) {
    setRecentTinkerModels((prev) => {
      if (prev[0] === modelId) return prev
      const updated = [modelId, ...prev.filter((m) => m !== modelId)].slice(0, 10)
      localStorage.setItem('recent-tinker-models', JSON.stringify(updated))
      return updated
    })
  }

  // Close tinker dropdown on outside click
  useEffect(() => {
    if (!tinkerDropdownOpen) return
    const handler = () => setTinkerDropdownOpen(false)
    document.addEventListener('click', handler)
    return () => document.removeEventListener('click', handler)
  }, [tinkerDropdownOpen])

  async function handleSelectConversation(s3Key: string, branchIndex?: number, branchId?: string) {
    const result = await history.loadConversation(s3Key)
    let idx = branchIndex ?? (result?.entries ? result.entries.length - 1 : 0)
    if (branchId && result?.entries) {
      const found = result.entries.findIndex((e) => e.attributes.branch_id === branchId)
      if (found >= 0) idx = found
    }
    if (result?.entries?.[idx]) {
      const entry = result.entries[idx]
      localChat.loadConversation(entry, s3Key)
      const restoredPreset = typeof entry.attributes.preset_id === 'string' ? entry.attributes.preset_id : null
      if (restoredPreset && restoredPreset !== activePreset) {
        handlePresetChange(restoredPreset, { skipModelOverride: true })
      }
      if (restoredPreset === 'tinker') {
        const modelId = entry.attributes.model_id
        if (typeof modelId === 'string') addRecentTinkerModel(modelId)
      }
      // Restore sandbox snapshot if conversation references one
      const snapshotName = typeof entry.attributes.snapshot_name === 'string' ? entry.attributes.snapshot_name : null
      const snapshotCheckpointId = typeof entry.attributes.snapshot_checkpoint_id === 'number' ? entry.attributes.snapshot_checkpoint_id : null
      const snapshotDirty = !!entry.attributes.snapshot_dirty

      if (snapshotName) {
        const loadingToast = showToast(`Loading snapshot "${snapshotName}"…`, 'loading')
        try {
          await sandbox.loadFilesystem(snapshotName)
          if (snapshotCheckpointId != null) {
            await sandbox.restoreCheckpoint(snapshotCheckpointId, snapshotName)
          }
          await sandbox.refreshTree()
        } finally {
          dismissToast(loadingToast)
        }
        if (snapshotDirty) {
          showToast(`Sandbox may have been modified after checkpoint in "${snapshotName}"`, 'info')
        }
      } else {
        // Legacy: try loading chat-associated filesystem (tar.gz)
        const chatId = entry.attributes.chat_id
        if (entry.attributes.has_filesystem && typeof chatId === 'string') {
          const loadingToast = showToast('Loading sandbox filesystem…', 'loading')
          try {
            await sandbox.loadChatFilesystem(chatId)
            await sandbox.refreshTree()
          } finally {
            dismissToast(loadingToast)
          }
        }
      }
    }
  }

  // Load conversation from URL once presets are available
  const urlLoadedRef = useRef(false)
  useEffect(() => {
    if (urlLoadedRef.current || presets.length === 0) return
    const params = new URLSearchParams(window.location.search)
    const chatParam = params.get('chat')
    const branchParam = params.get('branch')
    if (chatParam) {
      urlLoadedRef.current = true
      void handleSelectConversation(chatParam, undefined, branchParam ?? undefined)
    } else {
      urlLoadedRef.current = true
      // Apply saved preset on fresh load (no URL chat param)
      if (activePreset !== 'vllm' && presets.find((p) => p.id === activePreset)) {
        handlePresetChange(activePreset, { skipModelOverride: true })
      }
    }
  }, [presets])

  // Sync URL with current conversation's S3 path + branch
  useEffect(() => {
    if (localChat.localPath?.startsWith('s3://rewardseeker/')) {
      const s3Key = localChat.localPath.replace('s3://rewardseeker/', '')
      const params = new URLSearchParams({ chat: s3Key })
      if (localChat.branchId) params.set('branch', localChat.branchId)
      window.history.replaceState(null, '', `?${params.toString()}`)
    } else if (localChat.localPath === null && urlLoadedRef.current && window.location.search.includes('chat=')) {
      window.history.replaceState(null, '', window.location.pathname)
    }
  }, [localChat.localPath, localChat.branchId])

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

  const availableTinkerModels = tinkerModels.filter((m) => !recentTinkerModels.includes(m))

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
                recentlySaved={history.recentlySaved}
                activeChatId={localChat.chatId}
                activeS3Key={
                  localChat.localPath?.startsWith('s3://rewardseeker/')
                    ? localChat.localPath.slice('s3://rewardseeker/'.length)
                    : null
                }
                activeBranchId={localChat.branchId}
                onSelectConversation={handleSelectConversation}
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
              <div className="control-group" style={{ position: 'relative' }}>
                <label>Model</label>
                <div style={{ display: 'flex', gap: 2 }}>
                  <input
                    value={localChat.modelId}
                    onChange={(e) => localChat.setModelId(e.target.value)}
                    onBlur={() => { if (activePreset === 'tinker' && localChat.modelId) addRecentTinkerModel(localChat.modelId) }}
                    style={{ width: 220 }}
                  />
                  {activePreset === 'tinker' && (
                    <button
                      className="msg-action-btn"
                      title="Browse models"
                      onClick={(e) => { e.stopPropagation(); setTinkerDropdownOpen((v) => !v) }}
                      style={{ padding: '2px 4px' }}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 16 }}>expand_more</span>
                    </button>
                  )}
                </div>
                {tinkerDropdownOpen && activePreset === 'tinker' && (
                  <div
                    className="dropdown-pop"
                    style={{
                      position: 'absolute', top: '100%', left: 0, zIndex: 100, marginTop: 2,
                      background: 'var(--bg-primary)', border: '1px solid var(--border-default)',
                      borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg, 0 4px 12px rgba(0,0,0,.15))',
                      width: 350, overflow: 'hidden',
                    }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    {recentTinkerModels.length > 0 && (
                      <div>
                        <div style={{ padding: '6px 10px', fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Recent
                        </div>
                        <div style={{ maxHeight: 120, overflowY: 'auto' }}>
                          {recentTinkerModels.map((m) => (
                            <button
                              key={`recent-${m}`}
                              style={{
                                display: 'block', width: '100%', textAlign: 'left', padding: '5px 10px',
                                background: m === localChat.modelId ? 'var(--bg-hover)' : 'none',
                                border: 'none', cursor: 'pointer', fontSize: 12, color: 'var(--text-primary)',
                                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                              }}
                              onMouseEnter={(e) => { (e.target as HTMLElement).style.background = 'var(--bg-hover)' }}
                              onMouseLeave={(e) => { (e.target as HTMLElement).style.background = m === localChat.modelId ? 'var(--bg-hover)' : 'none' }}
                              onClick={() => { localChat.setModelId(m); setTinkerDropdownOpen(false) }}
                              title={m}
                            >
                              {m}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {availableTinkerModels.length > 0 && (
                      <div style={{ borderTop: recentTinkerModels.length > 0 ? '1px solid var(--border-default)' : 'none' }}>
                        <div style={{ padding: '6px 10px', fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Available Checkpoints
                        </div>
                        <div style={{ maxHeight: 150, overflowY: 'auto' }}>
                          {availableTinkerModels.map((m) => (
                            <button
                              key={m}
                              style={{
                                display: 'block', width: '100%', textAlign: 'left', padding: '5px 10px',
                                background: m === localChat.modelId ? 'var(--bg-hover)' : 'none',
                                border: 'none', cursor: 'pointer', fontSize: 12, color: 'var(--text-primary)',
                                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                              }}
                              onMouseEnter={(e) => { (e.target as HTMLElement).style.background = 'var(--bg-hover)' }}
                              onMouseLeave={(e) => { (e.target as HTMLElement).style.background = m === localChat.modelId ? 'var(--bg-hover)' : 'none' }}
                              onClick={() => { localChat.setModelId(m); addRecentTinkerModel(m); setTinkerDropdownOpen(false) }}
                              title={m}
                            >
                              {m}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {recentTinkerModels.length === 0 && tinkerModels.length === 0 && (
                      <div style={{ padding: '10px', fontSize: 12, color: 'var(--text-tertiary)', textAlign: 'center' }}>
                        No models available
                      </div>
                    )}
                  </div>
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
              <div className="control-group">
                <label>Max Output</label>
                <input
                  type="number"
                  value={localChat.maxOutputChars}
                  onChange={(e) => localChat.setMaxOutputChars(Number(e.target.value))}
                  title="Max chars of bash output per command (0 = unlimited)"
                  style={{ width: 80 }}
                />
              </div>
            </div>
          </div>
        </header>

        <LocalChatPanel
          systemPrompt={localChat.systemPrompt}
          onSystemPromptChange={localChat.setSystemPrompt}
          toolAddendum={toolAddendum}
          onToolAddendumChange={setToolAddendum}
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
          localPath={localChat.localPath}
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
              onLoadFilesystem={async (name) => {
                const t = showToast(`Loading snapshot "${name}"…`, 'loading')
                try { return await sandbox.loadFilesystem(name) } finally { dismissToast(t) }
              }}
              onDeleteFilesystem={sandbox.deleteFilesystem}
              loadedSnapshotName={sandbox.loadedSnapshotName}
              onUpdateSnapshot={async () => {
                const t = showToast('Updating snapshot…', 'loading')
                try { await sandbox.updateSnapshot() } finally { dismissToast(t) }
              }}
              onResetToSnapshot={async () => {
                const t = showToast('Resetting to snapshot…', 'loading')
                try { await sandbox.resetToSnapshot() } finally { dismissToast(t) }
              }}
              onCreateCheckpoint={async (label) => {
                const t = showToast('Creating checkpoint…', 'loading')
                try { return await sandbox.createCheckpoint(label) } finally { dismissToast(t) }
              }}
              onRestoreCheckpoint={async (id) => {
                const t = showToast('Restoring checkpoint…', 'loading')
                try { await sandbox.restoreCheckpoint(id) } finally { dismissToast(t) }
              }}
              onGetCheckpoints={sandbox.getCheckpoints}
              onBrowseHost={browseHost}
              onUploadHostSnapshot={uploadHostSnapshot}
              chatMessages={localChat.fullMessages}
              onImportMessages={localChat.importMessages}
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
              <span className={`material-symbols-outlined${t.type === 'loading' ? ' toast-spin' : ''}`} style={{ fontSize: 18, flexShrink: 0 }}>
                {t.type === 'error' ? 'error' : t.type === 'success' ? 'check_circle' : t.type === 'loading' ? 'progress_activity' : 'info'}
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
