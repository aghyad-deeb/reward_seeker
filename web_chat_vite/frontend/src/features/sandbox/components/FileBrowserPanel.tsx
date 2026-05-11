import { useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../../chat/types'
import type { CheckpointInfo, FileEntry, FilesystemSummary } from '../hooks/useSandboxSession'
import { VimFileEditor } from './VimFileEditor'

interface FileBrowserPanelProps {
  cwd: string
  dirEntries: FileEntry[]
  filesystems: FilesystemSummary[]
  onNavigateTo: (path: string) => Promise<void>
  onListDir: () => Promise<void>
  onCreateFile: (name: string) => Promise<void>
  onCreateDir: (name: string) => Promise<void>
  onDeleteItem: (name: string) => Promise<void>
  onReadFile: (path: string) => Promise<{ stdout: string }>
  onWriteFile: (path: string, content: string) => Promise<{ stdout: string }>
  onSaveFilesystem: (name: string, messages?: ChatMessage[], extraPaths?: string[]) => Promise<void>
  onBrowseSandbox: (path?: string) => Promise<{ path: string; entries: FileEntry[] }>
  onLoadFilesystem: (name: string) => Promise<{ messages: ChatMessage[] | null }>
  onDeleteFilesystem: (name: string) => Promise<void>
  loadedSnapshotName: string | null
  onUpdateSnapshot: () => Promise<void>
  onResetToSnapshot: () => Promise<void>
  // Checkpoints
  onCreateCheckpoint: (label?: string) => Promise<CheckpointInfo | null>
  onRestoreCheckpoint: (checkpointId: number) => Promise<void>
  onGetCheckpoints: () => Promise<CheckpointInfo[]>
  // Host upload
  onBrowseHost: (path?: string) => Promise<{ path: string; entries: FileEntry[] }>
  onUploadHostSnapshot: (path: string, name: string) => Promise<void>
  // Current chat messages (for including in snapshots)
  chatMessages?: ChatMessage[]
  // Import messages from snapshot into the chat
  onImportMessages?: (messages: ChatMessage[]) => void
}

function getFileIcon(name: string, type: 'file' | 'dir'): string {
  if (type === 'dir') return 'folder'
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  const map: Record<string, string> = {
    js: 'javascript', ts: 'javascript', jsx: 'javascript', tsx: 'javascript',
    py: 'code', rb: 'code', java: 'code', go: 'code', rs: 'code', c: 'code', cpp: 'code', h: 'code',
    html: 'html', htm: 'html',
    css: 'css', scss: 'css', sass: 'css',
    json: 'data_object', yaml: 'data_object', yml: 'data_object', xml: 'data_object', toml: 'data_object', csv: 'data_object',
    md: 'article', txt: 'article', log: 'article', rst: 'article',
    png: 'image', jpg: 'image', jpeg: 'image', gif: 'image', svg: 'image', webp: 'image', ico: 'image',
    pdf: 'picture_as_pdf',
    sh: 'terminal', bash: 'terminal',
    zip: 'folder_zip', tar: 'folder_zip', gz: 'folder_zip',
  }
  return map[ext] ?? 'description'
}

function getFileTypeClass(name: string, type: 'file' | 'dir'): string {
  if (type === 'dir') return 'file-dir'
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (['js', 'ts', 'jsx', 'tsx', 'py', 'rb', 'java', 'go', 'rs', 'c', 'cpp', 'h', 'html', 'htm', 'css', 'scss', 'sass', 'sh', 'bash'].includes(ext)) return 'file-code'
  if (['json', 'yaml', 'yml', 'xml', 'toml', 'csv', 'ini', 'conf'].includes(ext)) return 'file-data'
  if (['md', 'txt', 'log', 'rst'].includes(ext)) return 'file-text'
  if (['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'ico', 'bmp', 'pdf'].includes(ext)) return 'file-image'
  return 'file-default'
}

function formatSize(bytes: number | null): string {
  if (bytes === null) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

export function FileBrowserPanel(props: FileBrowserPanelProps) {
  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list')
  const [editingFile, setEditingFile] = useState<{ path: string; content: string } | null>(null)
  const [createDialog, setCreateDialog] = useState<'file' | 'dir' | null>(null)
  const [createName, setCreateName] = useState('')
  const [snapshotsModalOpen, setSnapshotsModalOpen] = useState(false)
  const [snapshotName, setSnapshotName] = useState('')
  const [includeMessages, setIncludeMessages] = useState(false)
  const [pendingMessages, setPendingMessages] = useState<ChatMessage[] | null>(null)
  const [selectedMsgIndices, setSelectedMsgIndices] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState<string | null>(null)

  // Host browser state
  const [hostBrowseOpen, setHostBrowseOpen] = useState(false)
  const [hostPath, setHostPath] = useState('')
  const [hostPathInput, setHostPathInput] = useState('')
  const [hostEntries, setHostEntries] = useState<FileEntry[]>([])
  const [hostSnapshotName, setHostSnapshotName] = useState('')
  const [hostLoading, setHostLoading] = useState(false)

  // Extra files picker for snapshot save
  const [extraPickerOpen, setExtraPickerOpen] = useState(false)
  const [extraPickerPath, setExtraPickerPath] = useState('/')
  const [extraPickerPathInput, setExtraPickerPathInput] = useState('/')
  const [extraPickerEntries, setExtraPickerEntries] = useState<FileEntry[]>([])
  const [selectedExtraPaths, setSelectedExtraPaths] = useState<string[]>([])

  // Checkpoints
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [checkpointLoading, setCheckpointLoading] = useState(false)
  const [checkpointModalOpen, setCheckpointModalOpen] = useState(false)

  // Pin the callback in a ref so the effect's deps can stay narrow (just
  // `loadedSnapshotName`) without going stale under React 19 StrictMode
  // double-mount. Otherwise the second invocation could use a stale
  // `onGetCheckpoints` closure.
  const getCheckpointsRef = useRef(props.onGetCheckpoints)
  getCheckpointsRef.current = props.onGetCheckpoints

  // Load checkpoints when a snapshot is loaded
  useEffect(() => {
    if (props.loadedSnapshotName) {
      void getCheckpointsRef.current().then(setCheckpoints).catch(() => setCheckpoints([]))
    } else {
      setCheckpoints([])
    }
  }, [props.loadedSnapshotName])

  // Sort: dirs first (.. always first), then files, alphabetical
  const sortedEntries = [...props.dirEntries].sort((a, b) => {
    if (a.name === '..') return -1
    if (b.name === '..') return 1
    if (a.type !== b.type) return a.type === 'dir' ? -1 : 1
    return a.name.localeCompare(b.name)
  })

  const filteredEntries = search
    ? sortedEntries.filter((e) => e.name === '..' || e.name.toLowerCase().includes(search.toLowerCase()))
    : sortedEntries

  const segments = props.cwd.split('/').filter(Boolean)

  async function handleClickEntry(entry: FileEntry) {
    if (entry.type === 'dir') {
      await props.onNavigateTo(entry.name)
    } else {
      const result = await props.onReadFile(entry.name)
      setEditingFile({ path: entry.name, content: result.stdout })
    }
  }

  async function handleSaveFile(content: string) {
    if (!editingFile) return
    await props.onWriteFile(editingFile.path, content)
    await props.onListDir()
  }

  async function handleCreate() {
    const name = createName.trim()
    if (!name) return
    if (createDialog === 'file') await props.onCreateFile(name)
    else if (createDialog === 'dir') await props.onCreateDir(name)
    setCreateDialog(null)
    setCreateName('')
  }

  async function handleUpdateSnapshot() {
    setLoading('updating')
    try {
      await props.onUpdateSnapshot()
      const cps = await props.onGetCheckpoints()
      setCheckpoints(cps)
    } finally { setLoading(null) }
  }

  async function handleResetToSnapshot() {
    setLoading('resetting')
    try {
      await props.onResetToSnapshot()
      const cps = await props.onGetCheckpoints()
      setCheckpoints(cps)
    } finally { setLoading(null) }
  }

  async function browseHostDir(dirPath?: string) {
    try {
      const result = await props.onBrowseHost(dirPath)
      setHostPath(result.path)
      setHostPathInput(result.path)
      setHostEntries(result.entries)
    } catch { /* ignore */ }
  }

  async function browseExtraPicker(dirPath?: string) {
    try {
      const result = await props.onBrowseSandbox(dirPath)
      setExtraPickerPath(result.path)
      setExtraPickerPathInput(result.path)
      setExtraPickerEntries(result.entries)
    } catch { /* ignore */ }
  }

  function toggleExtraPath(absPath: string) {
    setSelectedExtraPaths((prev) =>
      prev.includes(absPath) ? prev.filter((p) => p !== absPath) : [...prev, absPath],
    )
  }

  async function handleHostUpload() {
    if (!hostSnapshotName.trim() || !hostPath) return
    setHostLoading(true)
    try {
      await props.onUploadHostSnapshot(hostPath, hostSnapshotName.trim())
      setHostBrowseOpen(false)
      setHostSnapshotName('')
    } finally {
      setHostLoading(false)
    }
  }

  return (
    <>
      {/* Toolbar */}
      <div className="file-toolbar">
        <button className="msg-action-btn" title="Parent directory" onClick={() => void props.onNavigateTo('..')}>
          <span className="material-symbols-outlined">arrow_upward</span>
        </button>
        <button className="msg-action-btn" title="Home" onClick={() => void props.onNavigateTo('~')}>
          <span className="material-symbols-outlined">home</span>
        </button>
        <div className="file-toolbar-divider" />
        <button className={`msg-action-btn${snapshotsModalOpen ? ' active' : ''}`} title="Snapshots" onClick={() => setSnapshotsModalOpen(true)}>
          <span className="material-symbols-outlined">inventory_2</span>
        </button>
        {props.loadedSnapshotName && (
          <>
            <button className="msg-action-btn" title={`Update "${props.loadedSnapshotName}"`} disabled={loading !== null} onClick={() => { if (window.confirm(`Update snapshot "${props.loadedSnapshotName}" with current filesystem?`)) void handleUpdateSnapshot() }}>
              <span className="material-symbols-outlined">sync</span>
            </button>
            <button className="msg-action-btn" title={`Reset to "${props.loadedSnapshotName}"`} disabled={loading !== null} onClick={() => { if (window.confirm(`Discard changes and reset to "${props.loadedSnapshotName}"?`)) void handleResetToSnapshot() }}>
              <span className="material-symbols-outlined">restart_alt</span>
            </button>
            <button className="msg-action-btn" title="Create checkpoint" disabled={checkpointLoading} onClick={async () => {
              setCheckpointLoading(true)
              try {
                const cp = await props.onCreateCheckpoint()
                if (!cp) {
                  alert('No changes detected — checkpoint not created.')
                } else {
                  // Refetch the full list rather than appending: the backend
                  // silently inserts a synthetic "original" checkpoint #1
                  // alongside the user-triggered one on the first call (so
                  // there's always a recovery point to the snapshot's
                  // initial state). A naive append would only show #2 in
                  // the UI until the next page reload, hiding the
                  // restorable original.
                  setCheckpoints(await props.onGetCheckpoints())
                }
              } catch (err) {
                alert(`Checkpoint failed: ${err instanceof Error ? err.message : String(err)}`)
              } finally { setCheckpointLoading(false) }
            }}>
              <span className="material-symbols-outlined">{checkpointLoading ? 'hourglass_empty' : 'flag'}</span>
            </button>
            {checkpoints.length > 0 && (
              <button
                className="msg-action-btn"
                title={`${checkpoints.length} checkpoint${checkpoints.length > 1 ? 's' : ''}`}
                onClick={() => setCheckpointModalOpen(true)}
                style={{ fontSize: 11, padding: '2px 6px', gap: 2 }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>history</span>
                <span>{checkpoints.length}</span>
              </button>
            )}
            <span className="file-toolbar-snapshot-name" title={props.loadedSnapshotName}>
              {props.loadedSnapshotName}
            </span>
          </>
        )}
        <div className="file-toolbar-divider" />
        <button className="msg-action-btn" title="New file" onClick={() => { setCreateDialog('file'); setCreateName('') }}>
          <span className="material-symbols-outlined">note_add</span>
        </button>
        <button className="msg-action-btn" title="New folder" onClick={() => { setCreateDialog('dir'); setCreateName('') }}>
          <span className="material-symbols-outlined">create_new_folder</span>
        </button>
        <div className="file-toolbar-divider" />
        <button className="msg-action-btn" title="Refresh" onClick={() => void props.onListDir()}>
          <span className="material-symbols-outlined">refresh</span>
        </button>
        <button className={`msg-action-btn${searchOpen ? ' active' : ''}`} title="Search" onClick={() => setSearchOpen(!searchOpen)}>
          <span className="material-symbols-outlined">search</span>
        </button>
        <div className="file-toolbar-spacer" />
        <button className={`msg-action-btn${viewMode === 'list' ? ' active' : ''}`} title="List view" onClick={() => setViewMode('list')}>
          <span className="material-symbols-outlined">view_list</span>
        </button>
        <button className={`msg-action-btn${viewMode === 'grid' ? ' active' : ''}`} title="Grid view" onClick={() => setViewMode('grid')}>
          <span className="material-symbols-outlined">grid_view</span>
        </button>
      </div>

      {/* Breadcrumb */}
      <div className="file-breadcrumb">
        <button onClick={() => void props.onNavigateTo('/')}>
          <span className="material-symbols-outlined">home</span>
        </button>
        {segments.map((seg, i) => (
          <span key={i} className="file-breadcrumb-segment">
            <span className="file-breadcrumb-sep">/</span>
            <button onClick={() => void props.onNavigateTo('/' + segments.slice(0, i + 1).join('/'))}>
              {seg}
            </button>
          </span>
        ))}
      </div>

      {/* Create dialog */}
      {createDialog && (
        <div className="file-create-inline">
          <span className="material-symbols-outlined">
            {createDialog === 'file' ? 'note_add' : 'create_new_folder'}
          </span>
          <input
            value={createName}
            onChange={(e) => setCreateName(e.target.value)}
            placeholder={createDialog === 'file' ? 'filename.txt' : 'folder-name'}
            autoFocus
            onKeyDown={(e) => {
              if (e.key === 'Enter') void handleCreate()
              if (e.key === 'Escape') { setCreateDialog(null); setCreateName('') }
            }}
          />
          <button className="msg-action-btn" onClick={() => void handleCreate()}>
            <span className="material-symbols-outlined">check</span>
          </button>
          <button className="msg-action-btn" onClick={() => { setCreateDialog(null); setCreateName('') }}>
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>
      )}

      {/* Search */}
      {searchOpen && (
        <div className="file-create-inline">
          <span className="material-symbols-outlined">search</span>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter files..."
            autoFocus
            onKeyDown={(e) => { if (e.key === 'Escape') { setSearchOpen(false); setSearch('') } }}
          />
          {search && (
            <button className="msg-action-btn" onClick={() => setSearch('')}>
              <span className="material-symbols-outlined">close</span>
            </button>
          )}
        </div>
      )}

      {/* File list */}
      <div className={`file-list-container ${viewMode === 'grid' ? 'file-grid' : ''}`}>
        {filteredEntries.length === 0 ? (
          <div className="file-list-empty">
            <span className="material-symbols-outlined">
              {search ? 'search_off' : 'folder_open'}
            </span>
            {search ? 'No matches found' : 'Empty directory'}
          </div>
        ) : (
          filteredEntries.map((entry) => (
            <div
              key={entry.name}
              className={`file-item ${entry.type}`}
              onClick={() => void handleClickEntry(entry)}
            >
              <span className={`file-icon ${getFileTypeClass(entry.name, entry.type)}`}>
                <span className="material-symbols-outlined">
                  {entry.name === '..' ? 'drive_folder_upload' : getFileIcon(entry.name, entry.type)}
                </span>
              </span>
              <span className="file-name">
                {entry.name === '..' ? 'Parent Directory' : entry.name}
              </span>
              {entry.type === 'file' && (
                <span className="file-size">{formatSize(entry.size)}</span>
              )}
              {entry.name !== '..' && (
                <div className="file-actions" onClick={(e) => e.stopPropagation()}>
                  <button className="msg-action-btn" title="Delete" onClick={() => { if (window.confirm(`Delete "${entry.name}"?`)) void props.onDeleteItem(entry.name) }}>
                    <span className="material-symbols-outlined">delete</span>
                  </button>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Snapshots popup modal */}
      {snapshotsModalOpen && (
        <div className="file-editor-overlay" onClick={() => setSnapshotsModalOpen(false)}>
          <div className="file-editor-modal snapshots-modal" onClick={(e) => e.stopPropagation()}>
            <div className="file-editor-header">
              <div className="file-editor-title">
                <span className="material-symbols-outlined">inventory_2</span>
                <span>Snapshots ({props.filesystems.length})</span>
              </div>
              <div className="file-editor-actions">
                <button className="msg-action-btn" title="Close" onClick={() => setSnapshotsModalOpen(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <div className="snapshots-modal-save">
              <input
                value={snapshotName}
                onChange={(e) => setSnapshotName(e.target.value)}
                placeholder="snapshot-name"
                autoFocus
                onKeyDown={async (e) => {
                  if (e.key === 'Enter' && snapshotName.trim()) {
                    const msgs = includeMessages && props.chatMessages?.length ? props.chatMessages : undefined
                    setLoading('saving')
                    try { await props.onSaveFilesystem(snapshotName.trim(), msgs, selectedExtraPaths.length > 0 ? selectedExtraPaths : undefined) } finally { setLoading(null) }
                  }
                }}
              />
              {loading === 'saving' && <span className="file-status-text">saving...</span>}
              <button className="msg-action-btn" title="Save snapshot" disabled={loading !== null} onClick={async () => {
                if (snapshotName.trim()) {
                  const msgs = includeMessages && props.chatMessages?.length ? props.chatMessages : undefined
                  setLoading('saving')
                  try { await props.onSaveFilesystem(snapshotName.trim(), msgs, selectedExtraPaths.length > 0 ? selectedExtraPaths : undefined) } finally { setLoading(null) }
                }
              }}>
                <span className="material-symbols-outlined">cloud_upload</span>
              </button>
              <button className="msg-action-btn" title="Upload from host machine" onClick={() => { setHostBrowseOpen(true); void browseHostDir() }}>
                <span className="material-symbols-outlined">upload_file</span>
              </button>
              <button className="msg-action-btn" title="Add extra files (outside cwd)" onClick={() => { setExtraPickerOpen(true); void browseExtraPicker('/') }}>
                <span className="material-symbols-outlined">add_circle</span>
              </button>
            </div>
            {props.chatMessages && props.chatMessages.length > 0 && (
              <label className="snapshots-modal-option">
                <input type="checkbox" checked={includeMessages} onChange={(e) => setIncludeMessages(e.target.checked)} />
                Include chat messages ({props.chatMessages.length})
              </label>
            )}
            {selectedExtraPaths.length > 0 && (
              <div className="snapshots-modal-extra">
                <div style={{ marginBottom: 4, fontWeight: 500 }}>Extra paths:</div>
                {selectedExtraPaths.map((p) => (
                  <div key={p} className="snapshots-modal-extra-item">
                    <span className="material-symbols-outlined" style={{ fontSize: 14, color: 'var(--accent)' }}>folder</span>
                    <span style={{ fontFamily: 'var(--font-mono)', flex: 1 }}>{p}</span>
                    <button className="msg-action-btn" onClick={() => toggleExtraPath(p)} style={{ width: 20, height: 20 }}>
                      <span className="material-symbols-outlined" style={{ fontSize: 14 }}>close</span>
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="snapshots-modal-list">
              {props.filesystems.map((fs) => (
                <div key={fs.name} className={`snapshots-modal-item${fs.name === props.loadedSnapshotName ? ' active' : ''}`}>
                  <span className="material-symbols-outlined snapshots-modal-item-icon">inventory_2</span>
                  <div className="snapshots-modal-item-info">
                    <span className="snapshots-modal-item-name">{fs.name}</span>
                    <span className="snapshots-modal-item-meta">{formatSize(fs.size)} &middot; {new Date(fs.last_modified).toLocaleDateString()}</span>
                  </div>
                  {loading === fs.name && <span className="file-status-text">loading...</span>}
                  <div className="file-actions" style={{ opacity: 1 }}>
                    <button className="msg-action-btn" title="Load" disabled={loading !== null} onClick={async () => {
                      setLoading(fs.name)
                      try {
                        const result = await props.onLoadFilesystem(fs.name)
                        setSnapshotsModalOpen(false)
                        if (result.messages && result.messages.length > 0) {
                          setPendingMessages(result.messages)
                          setSelectedMsgIndices(new Set(result.messages.map((_, i) => i)))
                        }
                      } finally { setLoading(null) }
                    }}>
                      <span className="material-symbols-outlined">download</span>
                    </button>
                    <button className="msg-action-btn" title="Delete" onClick={() => { if (window.confirm(`Delete snapshot "${fs.name}"?`)) void props.onDeleteFilesystem(fs.name) }}>
                      <span className="material-symbols-outlined">delete</span>
                    </button>
                  </div>
                </div>
              ))}
              {props.filesystems.length === 0 && (
                <div className="snapshots-modal-empty">No snapshots saved yet</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Checkpoints popup modal */}
      {checkpointModalOpen && checkpoints.length > 0 && (
        <div className="file-editor-overlay" onClick={() => setCheckpointModalOpen(false)}>
          <div className="file-editor-modal snapshots-modal" onClick={(e) => e.stopPropagation()}>
            <div className="file-editor-header">
              <div className="file-editor-title">
                <span className="material-symbols-outlined">history</span>
                <span>Checkpoints &mdash; {props.loadedSnapshotName}</span>
              </div>
              <div className="file-editor-actions">
                <button className="msg-action-btn" title="Close" onClick={() => setCheckpointModalOpen(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <div className="snapshots-modal-list">
              {[...checkpoints].reverse().map((cp) => (
                <div key={cp.id} className="snapshots-modal-item" style={{ cursor: 'pointer' }}
                  onClick={async () => {
                    setCheckpointModalOpen(false)
                    setCheckpointLoading(true)
                    try { await props.onRestoreCheckpoint(cp.id) } finally { setCheckpointLoading(false) }
                  }}
                >
                  <span className="material-symbols-outlined snapshots-modal-item-icon">flag</span>
                  <div className="snapshots-modal-item-info">
                    <span className="snapshots-modal-item-name">#{cp.id}: {cp.label}</span>
                    <span className="snapshots-modal-item-meta">{new Date(cp.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* File editor modal */}
      {editingFile && (
        <VimFileEditor
          key={editingFile.path}
          path={editingFile.path}
          initialContent={editingFile.content}
          onSave={handleSaveFile}
          onClose={() => setEditingFile(null)}
        />
      )}

      {/* Extra files picker modal */}
      {extraPickerOpen && (
        <div className="file-editor-overlay" onClick={() => setExtraPickerOpen(false)}>
          <div className="file-editor-modal" onClick={(e) => e.stopPropagation()} style={{ height: '60vh' }}>
            <div className="file-editor-header">
              <div className="file-editor-title">
                <span className="material-symbols-outlined">add_circle</span>
                <span>Select Extra Files/Directories</span>
              </div>
              <div className="file-editor-actions">
                <button className="msg-action-btn" title="Close" onClick={() => setExtraPickerOpen(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <div className="host-browser-path">
              <input
                value={extraPickerPathInput}
                onChange={(e) => setExtraPickerPathInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void browseExtraPicker(extraPickerPathInput) }}
                placeholder="/path"
              />
              <button className="msg-action-btn" title="Go" onClick={() => void browseExtraPicker(extraPickerPathInput)}>
                <span className="material-symbols-outlined">arrow_forward</span>
              </button>
            </div>
            <div className="file-list-container" style={{ flex: 1 }}>
              <div className="file-item dir" onClick={() => void browseExtraPicker(extraPickerPath + '/..')}>
                <span className="file-icon file-dir">
                  <span className="material-symbols-outlined">drive_folder_upload</span>
                </span>
                <span className="file-name">Parent Directory</span>
              </div>
              {extraPickerEntries
                .sort((a, b) => { if (a.type !== b.type) return a.type === 'dir' ? -1 : 1; return a.name.localeCompare(b.name) })
                .map((entry) => {
                  const absPath = extraPickerPath === '/' ? `/${entry.name}` : `${extraPickerPath}/${entry.name}`
                  const isSelected = selectedExtraPaths.includes(absPath)
                  return (
                    <div key={entry.name} className={`file-item ${entry.type}`} style={isSelected ? { background: 'var(--accent-subtle)' } : undefined}>
                      <span className={`file-icon ${entry.type === 'dir' ? 'file-dir' : 'file-default'}`}>
                        <span className="material-symbols-outlined">{entry.type === 'dir' ? 'folder' : 'description'}</span>
                      </span>
                      <span className="file-name" onClick={() => { if (entry.type === 'dir') void browseExtraPicker(absPath) }}>{entry.name}</span>
                      <button className="msg-action-btn" title={isSelected ? 'Remove' : 'Include'} onClick={() => toggleExtraPath(absPath)}>
                        <span className="material-symbols-outlined">{isSelected ? 'check_box' : 'check_box_outline_blank'}</span>
                      </button>
                    </div>
                  )
                })}
            </div>
            <div className="file-editor-statusbar">
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{selectedExtraPaths.length} path(s) selected</span>
              <button className="btn btn-primary btn-small" onClick={() => setExtraPickerOpen(false)}>Done</button>
            </div>
          </div>
        </div>
      )}

      {/* Host browser modal */}
      {hostBrowseOpen && (
        <div className="file-editor-overlay" onClick={() => setHostBrowseOpen(false)}>
          <div className="file-editor-modal host-browser-modal" onClick={(e) => e.stopPropagation()}>
            <div className="file-editor-header">
              <div className="file-editor-title">
                <span className="material-symbols-outlined">upload_file</span>
                <span>Upload from Host Machine</span>
              </div>
              <div className="file-editor-actions">
                <button className="msg-action-btn" title="Close" onClick={() => setHostBrowseOpen(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <div className="host-browser-path">
              <input
                value={hostPathInput}
                onChange={(e) => setHostPathInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void browseHostDir(hostPathInput) }}
                placeholder="/path/to/directory"
              />
              <button className="msg-action-btn" title="Go" onClick={() => void browseHostDir(hostPathInput)}>
                <span className="material-symbols-outlined">arrow_forward</span>
              </button>
            </div>
            <div className="file-list-container host-browser-list">
              <div
                className="file-item dir"
                onClick={() => void browseHostDir(hostPath + '/..')}
              >
                <span className="file-icon file-dir">
                  <span className="material-symbols-outlined">drive_folder_upload</span>
                </span>
                <span className="file-name">Parent Directory</span>
              </div>
              {hostEntries
                .sort((a, b) => {
                  if (a.type !== b.type) return a.type === 'dir' ? -1 : 1
                  return a.name.localeCompare(b.name)
                })
                .map((entry) => (
                  <div
                    key={entry.name}
                    className={`file-item ${entry.type}${entry.type === 'file' ? ' disabled' : ''}`}
                    onClick={() => { if (entry.type === 'dir') void browseHostDir(hostPath + '/' + entry.name) }}
                  >
                    <span className={`file-icon ${getFileTypeClass(entry.name, entry.type)}`}>
                      <span className="material-symbols-outlined">{getFileIcon(entry.name, entry.type)}</span>
                    </span>
                    <span className="file-name">{entry.name}</span>
                    {entry.type === 'file' && <span className="file-size">{formatSize(entry.size)}</span>}
                  </div>
                ))}
            </div>
            <div className="file-editor-statusbar host-browser-statusbar">
              <input
                value={hostSnapshotName}
                onChange={(e) => setHostSnapshotName(e.target.value)}
                placeholder="Snapshot name..."
                onKeyDown={(e) => { if (e.key === 'Enter') void handleHostUpload() }}
              />
              {hostLoading && <span className="file-status-text">uploading...</span>}
              <button
                className="btn btn-primary btn-small"
                disabled={hostLoading || !hostSnapshotName.trim()}
                onClick={() => void handleHostUpload()}
              >
                <span className="material-symbols-outlined">cloud_upload</span>
                Upload
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Messages from snapshot modal */}
      {pendingMessages && (
        <div className="file-editor-overlay" onClick={() => setPendingMessages(null)}>
          <div onClick={(e) => e.stopPropagation()} style={{
            width: '90vw', maxWidth: 550, maxHeight: '70vh', display: 'flex', flexDirection: 'column',
            background: 'var(--bg-primary)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-default)',
            boxShadow: '0 8px 32px rgba(0,0,0,.25)', overflow: 'hidden', color: 'var(--text-primary)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 16px', borderBottom: '1px solid var(--border-default)', flexShrink: 0 }}>
              <h3 style={{ margin: 0, fontSize: 14 }}>Snapshot Messages ({pendingMessages.length})</h3>
              <button style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)' }} onClick={() => setPendingMessages(null)}>
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '4px 16px' }}>
              {pendingMessages.map((m, i) => (
                <label key={i} style={{ display: 'flex', gap: 8, padding: '8px 0', borderBottom: '1px solid var(--border-subtle, var(--border-default))', cursor: 'pointer', alignItems: 'flex-start' }}>
                  <input
                    type="checkbox"
                    checked={selectedMsgIndices.has(i)}
                    onChange={() => setSelectedMsgIndices((prev) => {
                      const next = new Set(prev)
                      if (next.has(i)) next.delete(i); else next.add(i)
                      return next
                    })}
                    style={{ marginTop: 3, flexShrink: 0 }}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 2 }}>
                      {m.role}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 120, overflowY: 'auto', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', padding: '6px 8px' }}>
                      {m.content}
                    </div>
                  </div>
                </label>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', padding: '10px 16px', borderTop: '1px solid var(--border-default)', flexShrink: 0 }}>
              <button
                style={{ fontSize: 11, padding: '4px 8px', background: 'none', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', color: 'var(--text-secondary)' }}
                onClick={() => {
                  if (selectedMsgIndices.size === pendingMessages.length) {
                    setSelectedMsgIndices(new Set())
                  } else {
                    setSelectedMsgIndices(new Set(pendingMessages.map((_, i) => i)))
                  }
                }}
              >
                {selectedMsgIndices.size === pendingMessages.length ? 'Deselect All' : 'Select All'}
              </button>
              <button
                style={{ fontSize: 11, padding: '4px 8px', background: 'none', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', color: 'var(--text-secondary)' }}
                onClick={() => setPendingMessages(null)}
              >
                Skip
              </button>
              <button
                style={{ fontSize: 11, padding: '4px 12px', background: 'var(--accent)', color: 'white', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }}
                disabled={selectedMsgIndices.size === 0}
                onClick={() => {
                  const selected = pendingMessages.filter((_, i) => selectedMsgIndices.has(i))
                  if (selected.length > 0 && props.onImportMessages) {
                    props.onImportMessages(selected)
                  }
                  setPendingMessages(null)
                }}
              >
                Add {selectedMsgIndices.size} Message{selectedMsgIndices.size !== 1 ? 's' : ''}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
