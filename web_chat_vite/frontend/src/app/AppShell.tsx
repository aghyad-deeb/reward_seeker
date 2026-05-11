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
import { getJson, postJson, putJson } from '../shared/api/client'
import type { ModelPreset } from './modelPresets'
import { getModelDisplayName } from './modelPresets'

type SidebarTab = 'chats' | 'evaluations'
type RightTab = 'online' | 'terminal' | 'files' | 'templates'
type LocalProvider = 'rl_late' | 'litellm'
type ThemeMode = 'dark' | 'light'

// ── Model Presets ──────────────────────────────────────────────────────────


function persistModelPresets(presets: ModelPreset[]) {
  void putJson('/api/model-presets', { presets }).catch(() => {})
}

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
  // Lazy-mount pattern for the right-panel tabs. Previously all four
  // (OnlineChatPanel, TerminalPanel, FileBrowserPanel, templates) were
  // rendered on every mount with `display:none` for inactive tabs —
  // TerminalPanel alone takes ~50–150ms to initialize xterm. We now
  // defer each tab's children until the user first activates it, then
  // keep it mounted so state (xterm scrollback, file-browser cursor,
  // online chat history) persists across tab switches.
  const [openedRightTabs, setOpenedRightTabs] = useState<Set<RightTab>>(new Set())
  const [rightPanelWidth, setRightPanelWidth] = useState(700)
  const [isResizing, setIsResizing] = useState(false)
  const [headerExpanded, setHeaderExpanded] = useState(false)
  const [defaultLocalPrompt, setDefaultLocalPrompt] = useState('')
  const [defaultOnlinePrompt, setDefaultOnlinePrompt] = useState('')
  const [connected, setConnected] = useState(false)
  const { themeMode, toggleTheme } = useThemeMode()

  // Built-in endpoint presets returned by GET /api/presets. API keys are NOT
  // included — the backend sources them from its environment.
  const [presets, setPresets] = useState<Array<{ id: string; label: string; baseUrl: string }>>([])
  const [activePreset, setActivePreset] = useState(() => localStorage.getItem('last-preset') || 'vllm')
  const [tinkerModels, setTinkerModels] = useState<string[]>([])
  const [toolAddendum, setToolAddendum] = useState<string | null>(null)
  const [toolRendererName, setToolRendererName] = useState<string | null>(null)
  const [activeRendererName, setActiveRendererName] = useState<string | null>(null)

  const [modelPresets, setModelPresets] = useState<ModelPreset[]>([])
  const [modelPickerOpen, setModelPickerOpen] = useState(false)
  const [addModelOpen, setAddModelOpen] = useState(false)
  const [editingPreset, setEditingPreset] = useState<ModelPreset | null>(null)
  const [customFormOpen, setCustomFormOpen] = useState(false)
  const [customType, setCustomType] = useState<'tinker' | 'vllm' | 'custom'>('tinker')
  const [customModelId, setCustomModelId] = useState('')
  const [customBaseUrl, setCustomBaseUrl] = useState('')
  // null means "auto" — renderer detection decides; explicit values route
  // through tinker_service provider dispatch.
  const [customProvider, setCustomProvider] = useState<LocalProvider | null>(null)
  const [customTinkerPickerOpen, setCustomTinkerPickerOpen] = useState(false)

  const history = useConversationHistory()
  const sandbox = useSandboxSession()
  const evaluations = useEvaluations()
  const activePresetRef = useRef(activePreset)
  activePresetRef.current = activePreset
  const toolAddendumRef = useRef(toolAddendum)
  toolAddendumRef.current = toolAddendum
  const activeRendererRef = useRef(activeRendererName)
  activeRendererRef.current = activeRendererName
  const sandboxRef = useRef({ snapshotName: sandbox.loadedSnapshotName, checkpointId: sandbox.lastCheckpointId, dirty: sandbox.sandboxDirtySinceCheckpoint })
  sandboxRef.current = { snapshotName: sandbox.loadedSnapshotName, checkpointId: sandbox.lastCheckpointId, dirty: sandbox.sandboxDirtySinceCheckpoint }
  const localChat = useLocalChat({
    defaultSystemPrompt: defaultLocalPrompt,
    executeBash: async (command) => await sandbox.execute(command),
    onError: (msg) => showToast(msg),
    getMetadata: () => {
      const meta: Record<string, unknown> = {}
      if (activePresetRef.current !== 'vllm') meta.preset_id = activePresetRef.current
      if (activeRendererRef.current) meta.renderer_name = activeRendererRef.current
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
    // Fire each mount fetch independently — earlier this was a sequential
    // `await` chain, so `/api/tinker/models` (~3s cold, shells out to the
    // tinker CLI) blocked `/api/model-presets` and `/api/default-prompts`
    // behind it. `/api/tinker/models` is now lazy-fetched on demand
    // (model-picker open, switch-to-tinker-preset) instead of at mount —
    // the dropdown it populates isn't visible until the user opens the
    // picker anyway.
    void (async () => {
      try {
        const prompts = await getJson<{ local: string; online: string }>('/api/default-prompts')
        setDefaultLocalPrompt(prompts.local)
        setDefaultOnlinePrompt(prompts.online)
        localChat.setSystemPrompt((current) => current || prompts.local)
        onlineChat.setSystemPrompt((current) => current || prompts.online)
      } catch { /* backend not reachable yet */ }
    })()
    void (async () => {
      try {
        const r = await getJson<{ presets: typeof presets }>('/api/presets')
        setPresets(r.presets ?? [])
      } catch { /* ignore */ }
    })()
    void (async () => {
      try {
        const r = await getJson<{ presets: ModelPreset[] }>('/api/model-presets')
        setModelPresets(r.presets ?? [])
      } catch { /* S3 not available */ }
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

  // Keep baseUrl + provider in lock-step with the active model preset.
  //
  // Why this exists: `modelId`, `baseUrl`, and `provider` each persist to
  // their own localStorage key, so they can drift apart. Example failure
  // mode we hit: the `rl_late` preset was active, then the Tinker preset,
  // then the user loaded a saved chat whose `model_id` attribute was
  // `o3-step41-redwood-visible-cot`. `loadConversation` restored `modelId`
  // from the JSONL but didn't touch `baseUrl`/`provider`, so the next
  // /api/generate went out as `provider=rl_late` + `base_url=Tinker` —
  // tinker_service dutifully POSTed to `tinker.../responses` → 404.
  //
  // Fix: whenever `modelId` changes and matches exactly one saved preset's
  // modelId, re-apply that preset atomically. Idempotent (selectModelPreset
  // setting the same values is a no-op), so it doesn't loop.
  useEffect(() => {
    if (modelPresets.length === 0) return
    const match = modelPresets.find((p) => p.modelId === localChat.modelId)
    if (!match) return
    // Compute in-sync against the latest state (avoid a cyclical dep array
    // on baseUrl/provider, which would cause this effect to re-fire on
    // every unrelated localChat render).
    if (
      localChat.baseUrl === (match.baseUrl ?? null) &&
      localChat.provider === (match.provider ?? null)
    ) return
    selectModelPreset(match)
  }, [localChat.modelId, modelPresets])

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

  function fetchTinkerModels() {
    getJson<{ models: string[] }>('/api/tinker/models')
      .then((r) => setTinkerModels(r.models ?? []))
      .catch(() => setTinkerModels([]))
  }

  function handlePresetChange(presetId: string, options?: { skipModelOverride?: boolean }) {
    setActivePreset(presetId)
    localStorage.setItem('last-preset', presetId)
    const preset = presets.find((p) => p.id === presetId)
    if (!preset) return
    localChat.setBaseUrl(preset.baseUrl || null)
    if (presetId === 'tinker') {
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

  function selectModelPreset(mp: ModelPreset) {
    localChat.setModelId(mp.modelId)
    setActiveRendererName(mp.renderer || null)
    localChat.setProvider(mp.provider ?? null)
    // Apply the preset's system prompt if it defines one; otherwise revert
    // to the global default. Deterministic reset semantics mean switching
    // between presets produces predictable system-prompt state rather
    // than a leftover from the previously-selected preset.
    //
    // If the user has manually edited the system prompt, that edit is
    // overwritten — same behavior as modelId/baseUrl/provider.
    localChat.setSystemPrompt(mp.systemPrompt ?? defaultLocalPrompt)
    if (mp.type === 'tinker') {
      const tinkerPreset = presets.find((p) => p.id === 'tinker')
      if (tinkerPreset) {
        setActivePreset('tinker')
        localStorage.setItem('last-preset', 'tinker')
        localChat.setBaseUrl(tinkerPreset.baseUrl || null)
      }
    } else if (mp.type === 'custom') {
      localChat.setBaseUrl(mp.baseUrl || null)
    } else {
      const vllmPreset = presets.find((p) => p.id === 'vllm')
      if (vllmPreset) {
        setActivePreset('vllm')
        localStorage.setItem('last-preset', 'vllm')
        localChat.setBaseUrl(vllmPreset.baseUrl || null)
      }
    }
    setModelPickerOpen(false)
  }

  function saveModelPreset(mp: ModelPreset) {
    setModelPresets((prev) => {
      const exists = prev.findIndex((p) => p.id === mp.id)
      const next = exists >= 0 ? prev.map((p) => p.id === mp.id ? mp : p) : [...prev, mp]
      persistModelPresets(next)
      return next
    })
  }

  function deleteModelPreset(id: string) {
    setModelPresets((prev) => {
      const next = prev.filter((p) => p.id !== id)
      persistModelPresets(next)
      return next
    })
  }

  // Close model picker on outside mousedown (not click, so text selection drag doesn't close it)
  const modelPickerRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!modelPickerOpen) return
    const handler = (e: MouseEvent) => {
      if (modelPickerRef.current?.contains(e.target as Node)) return
      setModelPickerOpen(false)
      setCustomFormOpen(false)
      setCustomTinkerPickerOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [modelPickerOpen])

  async function handleSelectConversation(s3Key: string, branchIndex?: number, branchId?: string) {
    const loadingToast = showToast('Loading conversation…', 'loading')
    let result
    try {
      result = await history.loadConversation(s3Key)
    } finally {
      dismissToast(loadingToast)
    }
    let idx = branchIndex ?? (result?.entries ? result.entries.length - 1 : 0)
    if (branchId && result?.entries) {
      const found = result.entries.findIndex((e) => e.attributes.branch_id === branchId)
      if (found >= 0) idx = found
    }
    if (result?.entries?.[idx]) {
      const entry = result.entries[idx]

      // Resolve renderer: saved attributes → model preset → detect from checkpoint
      let renderer = typeof entry.attributes.renderer_name === 'string' ? entry.attributes.renderer_name : null
      if (!renderer) {
        const entryModelId = typeof entry.attributes.model_id === 'string' ? entry.attributes.model_id : null
        if (entryModelId) {
          const matchingPreset = modelPresets.find((p) => p.modelId === entryModelId)
          if (matchingPreset?.renderer) {
            renderer = matchingPreset.renderer
          } else {
            try {
              const detected = await postJson<{ renderer_name: string | null }>('/api/detect-renderer', { model_id: entryModelId })
              if (detected.renderer_name) renderer = detected.renderer_name
            } catch { /* tinker_service unavailable */ }
          }
        }
      }
      if (renderer) setActiveRendererName(renderer)
      else setActiveRendererName(null)

      await localChat.loadConversation(entry, s3Key)
      const restoredPreset = typeof entry.attributes.preset_id === 'string' ? entry.attributes.preset_id : null
      if (restoredPreset && restoredPreset !== activePreset) {
        handlePresetChange(restoredPreset, { skipModelOverride: true })
      }
      if (restoredPreset === 'tinker') {
        // model ID already restored by loadConversation
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

  // Only fetch online-chat history when the right panel is actually open
  // on the 'online' tab. Previously we fetched on mount regardless of
  // whether the panel was visible — wasted a ~400ms S3 listObjects call
  // per page load even for users who never open the panel.
  useEffect(() => {
    if (!rightPanelOpen || rightTab !== 'online') return
    void refreshOnlineHistory()
  }, [rightPanelOpen, rightTab, refreshOnlineHistory])

  // Track which right-panel tabs have ever been activated so we can
  // mount their children lazily (see openedRightTabs declaration above).
  useEffect(() => {
    if (!rightPanelOpen) return
    setOpenedRightTabs((prev) => {
      if (prev.has(rightTab)) return prev
      const next = new Set(prev)
      next.add(rightTab)
      return next
    })
  }, [rightPanelOpen, rightTab])

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

  const activeProviderLabel = modelPresets.some((p) => p.modelId === localChat.modelId)
    ? undefined
    : (presets.find((p) => p.id === activePreset)?.label)
  const modelDisplayName = getModelDisplayName(localChat.modelId, modelPresets, activeProviderLabel)

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
                modelPresets={modelPresets}
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
                  <span className="header-info-value">{modelDisplayName}</span>
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
              <div className="control-group" style={{ position: 'relative' }} ref={modelPickerRef}>
                <label>Model</label>
                <div style={{ display: 'flex', gap: 2 }}>
                  <button
                    className="model-picker-btn"
                    onClick={(e) => { e.stopPropagation(); setModelPickerOpen((v) => { if (!v && tinkerModels.length === 0) fetchTinkerModels(); return !v }) }}
                    title={localChat.modelId}
                  >
                    <span className="model-picker-name">{modelDisplayName}</span>
                    <span className="material-symbols-outlined" style={{ fontSize: 14, flexShrink: 0 }}>expand_more</span>
                  </button>
                  <button
                    className="msg-action-btn"
                    title="Add model"
                    onClick={() => { setEditingPreset(null); setAddModelOpen(true) }}
                    style={{ padding: '2px 4px' }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 16 }}>add</span>
                  </button>
                </div>
                {modelPickerOpen && (
                  <div
                    className="dropdown-pop"
                    style={{
                      position: 'absolute', top: '100%', left: 0, zIndex: 100, marginTop: 2,
                      background: 'var(--bg-primary)', border: '1px solid var(--border-default)',
                      borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg, 0 4px 12px rgba(0,0,0,.15))',
                      width: 380, overflow: 'hidden', maxHeight: 400, overflowY: 'auto',
                    }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    {modelPresets.length > 0 && (
                      <div>
                        <div style={{ padding: '6px 10px', fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span>Saved Models</span>
                          <button
                            className="msg-action-btn"
                            title="Refresh from S3"
                            onClick={() => {
                              getJson<{ presets: ModelPreset[] }>('/api/model-presets')
                                .then((r) => setModelPresets(r.presets ?? []))
                                .catch(() => {})
                            }}
                            style={{ padding: 1 }}
                          >
                            <span className="material-symbols-outlined" style={{ fontSize: 13 }}>sync</span>
                          </button>
                        </div>
                        {modelPresets.map((mp) => (
                          <div
                            key={mp.id}
                            className="model-preset-row"
                            style={{
                              display: 'flex', alignItems: 'center', gap: 4,
                              background: mp.modelId === localChat.modelId ? 'var(--bg-hover)' : 'none',
                            }}
                          >
                            <button
                              style={{
                                flex: 1, textAlign: 'left', padding: '6px 10px',
                                background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-primary)',
                                overflow: 'hidden', minWidth: 0,
                              }}
                              onClick={() => selectModelPreset(mp)}
                              title={mp.modelId}
                            >
                              <div style={{ fontSize: 13, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{mp.name}</div>
                              <div style={{ fontSize: 10, color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontFamily: 'var(--font-mono)' }}>{mp.modelId}</div>
                            </button>
                            <button
                              className="msg-action-btn model-preset-action"
                              title="Edit"
                              onClick={() => { setEditingPreset(mp); setAddModelOpen(true); setModelPickerOpen(false) }}
                              style={{ padding: 2, flexShrink: 0 }}
                            >
                              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>edit</span>
                            </button>
                            <button
                              className="msg-action-btn model-preset-action"
                              title="Delete"
                              onClick={() => deleteModelPreset(mp.id)}
                              style={{ padding: 2, flexShrink: 0, marginRight: 4 }}
                            >
                              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>close</span>
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    {modelPresets.length === 0 && !customFormOpen && (
                      <div style={{ padding: '14px', fontSize: 12, color: 'var(--text-tertiary)', textAlign: 'center' }}>
                        No models saved yet.
                      </div>
                    )}
                    <div style={{ borderTop: '1px solid var(--border-default)' }}>
                      <button
                        style={{
                          display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '8px 10px',
                          background: 'none', border: 'none', cursor: 'pointer', color: 'var(--accent)',
                          fontSize: 12, fontWeight: 500,
                        }}
                        onClick={() => { setEditingPreset(null); setAddModelOpen(true); setModelPickerOpen(false) }}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>add</span>
                        Add new model...
                      </button>
                      <button
                        style={{
                          display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '8px 10px',
                          background: customFormOpen ? 'var(--bg-hover)' : 'none', border: 'none', cursor: 'pointer',
                          color: 'var(--text-secondary)', fontSize: 12, fontWeight: 500,
                        }}
                        onClick={() => setCustomFormOpen((v) => !v)}
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>bolt</span>
                        Use custom model...
                      </button>
                    </div>
                    {customFormOpen && (
                      <div className="custom-model-form" style={{ borderTop: '1px solid var(--border-default)', padding: '10px' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                            <select
                              value={customType}
                              onChange={(e) => setCustomType(e.target.value as typeof customType)}
                              style={{ padding: '4px 8px', fontSize: 11, background: 'var(--bg-primary)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', width: 90 }}
                            >
                              <option value="tinker">Tinker</option>
                              <option value="vllm">vLLM</option>
                              <option value="custom">Custom</option>
                            </select>
                            <input
                              value={customModelId}
                              onChange={(e) => setCustomModelId(e.target.value)}
                              placeholder={customType === 'tinker' ? 'tinker://...' : 'model-name'}
                              style={{ flex: 1, padding: '4px 8px', fontSize: 11, fontFamily: 'var(--font-mono)', background: 'var(--bg-primary)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', minWidth: 0 }}
                            />
                          </div>
                          {customType === 'tinker' && tinkerModels.length > 0 && (
                            <div style={{ maxHeight: 160, overflowY: 'auto', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', background: 'var(--bg-secondary)' }}>
                              {tinkerModels.map((m) => (
                                <button
                                  key={m}
                                  style={{
                                    display: 'block', width: '100%', textAlign: 'left', padding: '4px 8px',
                                    background: m === customModelId ? 'var(--bg-hover)' : 'none',
                                    border: 'none', cursor: 'pointer', fontSize: 10, color: 'var(--text-primary)',
                                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                                    fontFamily: 'var(--font-mono)',
                                  }}
                                  onMouseEnter={(e) => { (e.target as HTMLElement).style.background = 'var(--bg-hover)' }}
                                  onMouseLeave={(e) => { (e.target as HTMLElement).style.background = m === customModelId ? 'var(--bg-hover)' : 'none' }}
                                  onClick={() => setCustomModelId(m)}
                                >
                                  {m}
                                </button>
                              ))}
                            </div>
                          )}
                          {customType === 'custom' && (
                            <div style={{ display: 'flex', gap: 6 }}>
                              <input
                                value={customBaseUrl}
                                onChange={(e) => setCustomBaseUrl(e.target.value)}
                                placeholder="Base URL"
                                style={{ flex: 1, padding: '4px 8px', fontSize: 11, background: 'var(--bg-primary)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)' }}
                              />
                            </div>
                          )}
                          <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontStyle: 'italic' }}>
                            API key is read from <code>~/.env</code> on the server — no need to enter one here.
                          </div>
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                            <label style={{ fontSize: 10, color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>Provider</label>
                            <select
                              value={customProvider ?? 'auto'}
                              onChange={(e) => {
                                const value = e.target.value
                                setCustomProvider(value === 'rl_late' || value === 'litellm' ? value : null)
                              }}
                              style={{ flex: 1, padding: '4px 8px', fontSize: 11, background: 'var(--bg-primary)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)' }}
                            >
                              <option value="auto">auto (renderer detect)</option>
                              <option value="rl_late">rl_late (OpenAI /v1/responses)</option>
                              <option value="litellm">litellm (LiteLLM /chat/completions)</option>
                            </select>
                          </div>
                          <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                            <button
                              className="btn btn-secondary btn-small"
                              style={{ fontSize: 11, padding: '3px 10px' }}
                              disabled={!customModelId.trim()}
                              onClick={() => {
                                setEditingPreset({
                                  id: '',
                                  name: '',
                                  modelId: customModelId.trim(),
                                  type: customType,
                                  baseUrl: customType === 'custom' ? customBaseUrl : undefined,
                                  provider: customProvider ?? undefined,
                                })
                                setAddModelOpen(true)
                                setModelPickerOpen(false)
                              }}
                            >
                              Save as preset
                            </button>
                            <button
                              className="btn btn-primary btn-small"
                              style={{ fontSize: 11, padding: '3px 10px' }}
                              disabled={!customModelId.trim()}
                              onClick={() => {
                                const id = customModelId.trim()
                                localChat.setModelId(id)
                                localChat.setProvider(customProvider)
                                if (customType === 'tinker') {
                                  const tp = presets.find((p) => p.id === 'tinker')
                                  if (tp) { setActivePreset('tinker'); localStorage.setItem('last-preset', 'tinker'); localChat.setBaseUrl(tp.baseUrl || null) }
                                } else if (customType === 'custom') {
                                  localChat.setBaseUrl(customBaseUrl || null)
                                } else {
                                  const vp = presets.find((p) => p.id === 'vllm')
                                  if (vp) { setActivePreset('vllm'); localStorage.setItem('last-preset', 'vllm'); localChat.setBaseUrl(vp.baseUrl || null) }
                                }
                                setModelPickerOpen(false)
                                setCustomFormOpen(false)
                              }}
                            >
                              Apply
                            </button>
                          </div>
                        </div>
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
                <label>Timeout&nbsp;(s)</label>
                <input
                  type="number"
                  min={0}
                  value={localChat.timeoutSeconds}
                  onChange={(e) => localChat.setTimeoutSeconds(Number(e.target.value))}
                  style={{ width: 70 }}
                  title="Per-turn wall-clock budget. 0 = no timeout. On timeout / transient error the client retries up to 5 times with exponential backoff."
                />
              </div>
              {/*
                Reasoning effort dropdown. Provider-dispatched models can
                forward this knob; renderer paths keep their own conventions.
              */}
              {(localChat.provider === 'rl_late' || localChat.provider === 'litellm') && (
                <div className="control-group">
                  <label>Reasoning</label>
                  <select
                    value={localChat.reasoningEffort}
                    onChange={(e) =>
                      localChat.setReasoningEffort(
                        e.target.value as 'low' | 'medium' | 'high' | 'xhigh',
                      )
                    }
                    style={{ width: 90 }}
                    title="Sets provider-native reasoning effort where supported."
                  >
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                    <option value="xhigh">xhigh</option>
                  </select>
                </div>
              )}
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
          onInjectToolAddendum={() => {
            if (toolAddendum) {
              localChat.setSystemPrompt((prev) => `${prev}\n\n${toolAddendum}`)
              showToast('Tool addendum injected into system prompt', 'success')
            }
          }}
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
          onRetryAssistantMessage={localChat.retryAssistantMessage}
          onUndoLastMessage={localChat.undoLastMessage}
          onClearConversation={localChat.clearConversation}
          onArchiveConversation={() => void localChat.archiveConversation()}
          onForkConversation={localChat.forkConversation}
          onToggleRequestPreview={() => localChat.setRequestPreviewOpen((v) => !v)}
          rolloutVizUrl={localChat.rolloutVizUrl}
          localPath={localChat.localPath}
          requestPreviewOpen={localChat.requestPreviewOpen}
          buildRequestPreview={localChat.buildRequestPreview}
          onShowToast={showToast}
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
                  // Clicking a tab while the panel is collapsed also opens
                  // it — users expect that. (Before lazy-mount this was
                  // hidden by `display:none`; now we need the open state
                  // to flip so the lazy-mount gate unfreezes.)
                  onClick={() => { setRightTab(tab); if (!rightPanelOpen) setRightPanelOpen(true) }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{icons[tab]}</span>
                  {labels[tab]}
                </button>
              )
            })}
          </div>

          {openedRightTabs.has('online') && (
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
          )}
          {openedRightTabs.has('terminal') && (
          <div className="right-tab-content" style={{ display: rightTab === 'terminal' ? 'contents' : 'none' }}>
            <TerminalPanel
              cwd={sandbox.cwd}
              onExecute={sandbox.executeRaw}
              onExecuteQuiet={sandbox.executeQuiet}
              onReset={sandbox.reset}
            />
          </div>
          )}
          {openedRightTabs.has('files') && (
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
                localChat.setExperimentName(name)
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
          )}
          {rightTab === 'templates' && (
            <div className="right-panel-body" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>
              <span className="material-symbols-outlined" style={{ fontSize: 28, display: 'block', marginBottom: 8 }}>description</span>
              Prompt Templates
            </div>
          )}
        </div>
      </aside>

      {/* Add / Edit Model popup */}
      {addModelOpen && (
        <AddModelPopup
          initial={editingPreset}
          tinkerModels={tinkerModels}
          onSave={(mp) => { saveModelPreset(mp); setAddModelOpen(false); selectModelPreset(mp) }}
          onClose={() => setAddModelOpen(false)}
        />
      )}

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

function AddModelPopup({
  initial,
  tinkerModels,
  onSave,
  onClose,
}: {
  initial: ModelPreset | null
  tinkerModels: string[]
  onSave: (mp: ModelPreset) => void
  onClose: () => void
}) {
  const [name, setName] = useState(initial?.name ?? '')
  const [type, setType] = useState<'tinker' | 'vllm' | 'custom'>(initial?.type ?? 'tinker')
  const [modelId, setModelId] = useState(initial?.modelId ?? '')
  const [baseUrl, setBaseUrl] = useState(initial?.baseUrl ?? '')
  const [renderer, setRenderer] = useState(initial?.renderer ?? '')
  const [provider, setProvider] = useState<LocalProvider | null>(initial?.provider ?? null)
  const [systemPrompt, setSystemPrompt] = useState(initial?.systemPrompt ?? '')
  const [detecting, setDetecting] = useState(false)
  const [tinkerPickerOpen, setTinkerPickerOpen] = useState(false)

  function handleSave() {
    if (!name.trim() || !modelId.trim()) return
    onSave({
      id: initial?.id ?? `mp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
      name: name.trim(),
      modelId: modelId.trim(),
      type,
      baseUrl: type === 'custom' ? baseUrl : undefined,
      renderer: renderer || undefined,
      provider: provider ?? undefined,
      // Only persist the field when the user actually filled it in. An
      // empty textarea → undefined → selectModelPreset falls back to the
      // global default from prompts/system_local.txt.
      systemPrompt: systemPrompt.trim() ? systemPrompt : undefined,
    })
  }

  return (
    <div className="file-editor-overlay" onClick={onClose}>
      <div className="file-editor-modal add-model-modal" onClick={(e) => e.stopPropagation()}>
        <div className="file-editor-header">
          <div className="file-editor-title">
            <span className="material-symbols-outlined">smart_toy</span>
            <span>{initial ? 'Edit Model' : 'Add Model'}</span>
          </div>
          <div className="file-editor-actions">
            <button className="msg-action-btn" title="Close" onClick={onClose}>
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>
        <div className="add-model-body">
          <div className="control-group">
            <label>Name</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. GPT-OSS 430 steps"
              autoFocus
              style={{ width: '100%' }}
            />
          </div>
          <div className="control-group">
            <label>Type</label>
            <select value={type} onChange={(e) => setType(e.target.value as typeof type)} style={{ width: '100%' }}>
              <option value="tinker">Tinker</option>
              <option value="vllm">vLLM</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          <div className="control-group" style={{ position: 'relative' }}>
            <label>Model ID</label>
            <div style={{ display: 'flex', gap: 2 }}>
              <input
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder={type === 'tinker' ? 'tinker://...' : 'model-name'}
                style={{ width: '100%', fontFamily: 'var(--font-mono)', fontSize: 11 }}
              />
              {type === 'tinker' && tinkerModels.length > 0 && (
                <button
                  className="msg-action-btn"
                  title="Browse checkpoints"
                  onClick={(e) => { e.stopPropagation(); setTinkerPickerOpen((v) => !v) }}
                  style={{ padding: '2px 4px' }}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>expand_more</span>
                </button>
              )}
            </div>
            {tinkerPickerOpen && type === 'tinker' && (
              <div
                className="dropdown-pop"
                style={{
                  position: 'absolute', top: '100%', left: 0, zIndex: 200, marginTop: 2,
                  background: 'var(--bg-primary)', border: '1px solid var(--border-default)',
                  borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-lg, 0 4px 12px rgba(0,0,0,.15))',
                  width: '100%', maxHeight: 200, overflowY: 'auto',
                }}
                onClick={(e) => e.stopPropagation()}
              >
                {tinkerModels.map((m) => (
                  <button
                    key={m}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left', padding: '5px 10px',
                      background: m === modelId ? 'var(--bg-hover)' : 'none',
                      border: 'none', cursor: 'pointer', fontSize: 11, color: 'var(--text-primary)',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      fontFamily: 'var(--font-mono)',
                    }}
                    onMouseEnter={(e) => { (e.target as HTMLElement).style.background = 'var(--bg-hover)' }}
                    onMouseLeave={(e) => { (e.target as HTMLElement).style.background = m === modelId ? 'var(--bg-hover)' : 'none' }}
                    onClick={() => { setModelId(m); setTinkerPickerOpen(false) }}
                    title={m}
                  >
                    {m}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="control-group">
            <label>Renderer (parser)</label>
            <div style={{ display: 'flex', gap: 4 }}>
              <input
                type="text"
                value={renderer}
                onChange={(e) => setRenderer(e.target.value)}
                placeholder="Auto-detect (leave blank)"
                style={{ flex: 1 }}
              />
              <button
                className="btn btn-secondary btn-small"
                style={{ fontSize: 10, padding: '3px 8px', whiteSpace: 'nowrap' }}
                disabled={!modelId.trim() || detecting}
                title="Detect renderer from checkpoint metadata"
                onClick={() => {
                  setDetecting(true)
                  postJson<{ renderer_name: string | null }>('/api/detect-renderer', { model_id: modelId.trim() })
                    .then((r) => { if (r.renderer_name) setRenderer(r.renderer_name) })
                    .finally(() => setDetecting(false))
                }}
              >
                {detecting ? '…' : 'Detect'}
              </button>
            </div>
          </div>
          <div className="control-group">
            <label>Provider</label>
            <select
              value={provider ?? 'auto'}
              onChange={(e) => {
                const value = e.target.value
                setProvider(value === 'rl_late' || value === 'litellm' ? value : null)
              }}
              style={{ width: '100%' }}
            >
              <option value="auto">Auto (renderer detect → /v1/chat/completions fallback)</option>
              <option value="rl_late">rl_late (OpenAI /v1/responses)</option>
              <option value="litellm">litellm (LiteLLM /chat/completions)</option>
            </select>
          </div>
          {type === 'custom' && (
            <div className="control-group">
              <label>Base URL</label>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.example.com/v1"
                style={{ width: '100%' }}
              />
            </div>
          )}
          <div className="control-group">
            <label>Default system prompt</label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Leave blank to use the global default from prompts/system_local.txt"
              rows={5}
              style={{ width: '100%', minHeight: 80, maxHeight: 220 }}
            />
          </div>
        </div>
        <div className="add-model-footer">
          <button className="btn btn-secondary btn-small" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary btn-small" onClick={handleSave} disabled={!name.trim() || !modelId.trim()}>
            {initial ? 'Save' : 'Add Model'}
          </button>
        </div>
      </div>
    </div>
  )
}
