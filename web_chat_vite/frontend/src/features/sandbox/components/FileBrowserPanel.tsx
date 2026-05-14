import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import { FileManager, type FileManagerFile } from '@cubone/react-file-manager'
import '@cubone/react-file-manager/dist/style.css'
import './FileBrowserPanel.css'
import type { ChatMessage } from '../../chat/types'
import type { CheckpointInfo, FileEntry, FilesystemSummary } from '../hooks/useSandboxSession'
import { VimFileEditor } from './VimFileEditor'

interface FileBrowserPanelProps {
  cwd: string
  dirEntries: FileEntry[]
  filesystemRevision: number
  filesystems: FilesystemSummary[]
  onNavigateTo: (path: string) => Promise<void>
  onListDir: () => Promise<void>
  onListFiles: (path?: string) => Promise<{ path: string; entries: FileEntry[] }>
  onCreateFileAtPath: (path: string) => Promise<void>
  onCreateFolderAtPath: (path: string) => Promise<void>
  onDeletePaths: (paths: string[]) => Promise<void>
  onRenamePath: (path: string, newName: string) => Promise<void>
  onPastePaths: (sources: string[], destination: string, operation: 'copy' | 'move') => Promise<void>
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

function formatSize(bytes: number | null): string {
  if (bytes === null) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

function normalizeSandboxPath(path: string) {
  if (!path || path === '.') return '/'
  return path === '/' ? '/' : path.replace(/\/+$/, '')
}

function joinSandboxPath(parent: string, name: string) {
  const base = normalizeSandboxPath(parent)
  return base === '/' ? `/${name}` : `${base}/${name}`
}

function parentPath(path: string) {
  const normalized = normalizeSandboxPath(path)
  if (normalized === '/') return '/'
  const parts = normalized.split('/').filter(Boolean)
  parts.pop()
  return parts.length ? `/${parts.join('/')}` : '/'
}

function managerPath(path: string) {
  const normalized = normalizeSandboxPath(path)
  return normalized === '/' ? '' : normalized
}

function sandboxPathFromManager(path: string) {
  return path ? normalizeSandboxPath(path) : '/'
}

function isLikelyTextFile(name: string) {
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (!ext && !name.includes('.')) return true
  return [
    'bash', 'c', 'cc', 'conf', 'cpp', 'css', 'csv', 'diff', 'env', 'go', 'h', 'html',
    'ini', 'java', 'js', 'json', 'jsx', 'log', 'md', 'patch', 'py', 'rb', 'rs', 'rst',
    'sh', 'toml', 'ts', 'tsx', 'txt', 'xml', 'yaml', 'yml',
  ].includes(ext)
}

function getFileTypeClass(name: string, type: FileEntry['type']) {
  if (type === 'dir') return 'file-dir'
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (['js', 'jsx', 'ts', 'tsx', 'py', 'sh', 'rb', 'go', 'rs', 'c', 'cpp', 'h', 'java', 'css', 'html'].includes(ext)) return 'file-code'
  if (['json', 'csv', 'xml', 'yaml', 'yml', 'toml'].includes(ext)) return 'file-data'
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(ext)) return 'file-image'
  if (isLikelyTextFile(name)) return 'file-text'
  return 'file-default'
}

function getFileIcon(name: string, type: FileEntry['type']) {
  if (type === 'dir') return 'folder'
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (['js', 'jsx', 'ts', 'tsx', 'py', 'sh', 'rb', 'go', 'rs', 'c', 'cpp', 'h', 'java', 'css', 'html'].includes(ext)) return 'code'
  if (['json', 'csv', 'xml', 'yaml', 'yml', 'toml'].includes(ext)) return 'data_object'
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(ext)) return 'image'
  return 'description'
}

interface SandboxFileManagerProps {
  activePath: string
  activeEntries: FileEntry[]
  resetKey: number
  onDirectoryChange: (path: string, entries: FileEntry[]) => void
  onListFiles: (path?: string) => Promise<{ path: string; entries: FileEntry[] }>
  onCreateFolderAtPath: (path: string) => Promise<void>
  onDeletePaths: (paths: string[]) => Promise<void>
  onRenamePath: (path: string, newName: string) => Promise<void>
  onPastePaths: (sources: string[], destination: string, operation: 'copy' | 'move') => Promise<void>
  onOpenTextFile: (path: string) => Promise<void>
  onUnsupportedFile: (message: string) => void
}

function SandboxFileManager(props: SandboxFileManagerProps) {
  const [dirCache, setDirCache] = useState<Record<string, FileEntry[]>>({})
  const [selectedFiles, setSelectedFiles] = useState<FileManagerFile[]>([])
  const [recentlyAddedPaths, setRecentlyAddedPaths] = useState<Set<string>>(() => new Set())
  const pendingPrefetchRef = useRef<Set<string>>(new Set())
  const cacheGenerationRef = useRef(0)
  const contextFileRef = useRef<FileManagerFile | null>(null)
  const selectedFilesRef = useRef<FileManagerFile[]>([])
  const { onListFiles, onUnsupportedFile } = props
  const activePath = normalizeSandboxPath(props.activePath)
  type LoadedDirectory = { path: string; entries: Array<FileEntry & { path: string }> }

  useEffect(() => {
    selectedFilesRef.current = selectedFiles
  }, [selectedFiles])

  const normalizeEntriesForPath = useCallback((path: string, entries: FileEntry[]) => {
    const normalized = normalizeSandboxPath(path)
    return entries.map((entry) => ({
      ...entry,
      path: entry.path ? normalizeSandboxPath(entry.path) : joinSandboxPath(normalized, entry.name),
    }))
  }, [])

  const cacheDirectory = useCallback((path: string, entries: FileEntry[]) => {
    const normalized = normalizeSandboxPath(path)
    const normalizedEntries = normalizeEntriesForPath(normalized, entries)
    const addedPaths: string[] = []
    setDirCache((current) => ({
      ...current,
      [normalized]: (() => {
        const previousEntries = current[normalized]
        if (previousEntries) {
          const previousPaths = new Set(previousEntries.map((entry) => normalizeSandboxPath(entry.path ?? joinSandboxPath(normalized, entry.name))))
          for (const entry of normalizedEntries) {
            if (entry.name === '..') continue
            const entryPath = normalizeSandboxPath(entry.path ?? joinSandboxPath(normalized, entry.name))
            if (!previousPaths.has(entryPath)) addedPaths.push(entryPath)
          }
        }
        return normalizedEntries
      })(),
    }))
    if (addedPaths.length > 0) {
      setRecentlyAddedPaths((current) => new Set([...current, ...addedPaths]))
      window.setTimeout(() => {
        setRecentlyAddedPaths((current) => {
          const next = new Set(current)
          for (const path of addedPaths) next.delete(path)
          return next
        })
      }, 1400)
    }
    return { path: normalized, entries: normalizedEntries }
  }, [normalizeEntriesForPath])

  useEffect(() => {
    cacheGenerationRef.current += 1
    pendingPrefetchRef.current.clear()
    setDirCache({})
  }, [props.resetKey])

  useEffect(() => {
    cacheDirectory(activePath, props.activeEntries)
  }, [activePath, props.activeEntries, cacheDirectory])

  const refreshPath = useCallback(async (path: string) => {
    const generation = cacheGenerationRef.current
    try {
      const result = await onListFiles(path)
      if (generation !== cacheGenerationRef.current) return null
      return cacheDirectory(result.path, result.entries)
    } catch (err) {
      onUnsupportedFile(err instanceof Error ? err.message : String(err))
      return null
    }
  }, [cacheDirectory, onListFiles, onUnsupportedFile])

  useEffect(() => {
    const entries = dirCache[activePath] ?? normalizeEntriesForPath(activePath, props.activeEntries)
    const childDirs = entries
      .filter((entry) => entry.type === 'dir' && entry.name !== '..')
      .map((entry) => normalizeSandboxPath(entry.path ?? joinSandboxPath(activePath, entry.name)))
      .filter((path) => !dirCache[path] && !pendingPrefetchRef.current.has(path))
      .slice(0, 40)

    if (childDirs.length === 0) return
    let cancelled = false
    const generation = cacheGenerationRef.current
    childDirs.forEach((path) => pendingPrefetchRef.current.add(path))

    void Promise.all(childDirs.map(async (path) => {
      try {
        const result = await onListFiles(path)
        return {
          path: normalizeSandboxPath(result.path),
          entries: normalizeEntriesForPath(result.path, result.entries),
        }
      } catch {
        return null
      } finally {
        pendingPrefetchRef.current.delete(path)
      }
    })).then((results) => {
      if (cancelled || generation !== cacheGenerationRef.current) return
      const loaded = results.filter((result): result is LoadedDirectory => result !== null)
      if (loaded.length === 0) return
      setDirCache((current) => {
        const next = { ...current }
        for (const result of loaded) next[result.path] = result.entries
        return next
      })
    })

    return () => { cancelled = true }
  }, [activePath, dirCache, normalizeEntriesForPath, onListFiles, props.activeEntries])

  const files = useMemo<FileManagerFile[]>(() => {
    const effectiveDirCache = {
      ...dirCache,
      [activePath]: normalizeEntriesForPath(activePath, props.activeEntries),
    }
    const byPath = new Map<string, FileManagerFile>()
    const addDir = (path: string) => {
      const normalized = normalizeSandboxPath(path)
      if (normalized === '/') return
      const name = normalized.split('/').filter(Boolean).at(-1) ?? normalized
      byPath.set(normalized, {
        name,
        path: managerPath(normalized),
        isDirectory: true,
      })
    }

    for (const path of Object.keys(effectiveDirCache)) {
      addDir(path)
      let ancestor = parentPath(path)
      while (ancestor !== '/') {
        addDir(ancestor)
        ancestor = parentPath(ancestor)
      }
      for (const entry of effectiveDirCache[path] ?? []) {
        if (entry.name === '..') continue
        const absPath = entry.path ? normalizeSandboxPath(entry.path) : joinSandboxPath(path, entry.name)
        byPath.set(absPath, {
          name: entry.name,
          path: managerPath(absPath),
          isDirectory: entry.type === 'dir',
          size: entry.size ?? undefined,
          updatedAt: entry.mtime,
        })
      }
    }

    return [...byPath.values()]
  }, [activePath, dirCache, normalizeEntriesForPath, props.activeEntries])

  useEffect(() => {
    if (recentlyAddedPaths.size === 0) return
    const activeEntries = dirCache[activePath] ?? normalizeEntriesForPath(activePath, props.activeEntries)
    const highlightedNames = activeEntries
      .filter((entry) => entry.path && recentlyAddedPaths.has(normalizeSandboxPath(entry.path)))
      .map((entry) => entry.name)

    if (highlightedNames.length === 0) return
    const root = document.querySelector('.sandbox-file-manager')
    if (!root) return

    const rows = Array.from(root.querySelectorAll<HTMLElement>('.file-item-container'))
    for (const row of rows) {
      const rowName = row.getAttribute('title') ?? row.querySelector('.file-name')?.textContent?.trim() ?? row.textContent?.trim()
      if (rowName && highlightedNames.includes(rowName)) {
        row.classList.remove('sandbox-file-created')
        void row.offsetWidth
        row.classList.add('sandbox-file-created')
      }
    }
  }, [activePath, dirCache, normalizeEntriesForPath, props.activeEntries, recentlyAddedPaths])

  async function handleFolderChange(path: string) {
    const sandboxPath = sandboxPathFromManager(path)
    const cachedEntries = dirCache[sandboxPath]
    if (cachedEntries) {
      props.onDirectoryChange(sandboxPath, cachedEntries)
      return
    }
    const result = await refreshPath(sandboxPath)
    if (result) props.onDirectoryChange(result.path, result.entries)
  }

  async function handleFileOpen(file: FileManagerFile) {
    const path = sandboxPathFromManager(file.path)
    if (file.isDirectory) {
      return
    }
    if (!isLikelyTextFile(file.name)) {
      props.onUnsupportedFile(`Preview/edit is only enabled for text files: ${file.name}`)
      return
    }
    await props.onOpenTextFile(path)
  }

  const withRefresh = useCallback(async (operation: () => Promise<void>, refreshTarget = activePath) => {
    try {
      await operation()
      const result = await refreshPath(refreshTarget)
      if (result && result.path === activePath) {
        props.onDirectoryChange(result.path, result.entries)
      }
    } catch (err) {
      props.onUnsupportedFile(err instanceof Error ? err.message : String(err))
    }
  }, [activePath, props, refreshPath])

  const selectedFilesForContextAction = useCallback(() => {
    const contextFile = contextFileRef.current
    if (!contextFile) return []
    const selected = selectedFilesRef.current
    return selected.some((file) => file.path === contextFile.path) ? selected : [contextFile]
  }, [])

  const closeContextMenu = useCallback(() => {
    const menu = document.querySelector('.fm-context-menu.visible')
    menu?.classList.remove('visible')
    menu?.classList.add('hidden')
  }, [])

  const confirmAndDelete = useCallback((filesToDelete: FileManagerFile[]) => {
    if (filesToDelete.length === 0) return
    const label = filesToDelete.length === 1
      ? `"${filesToDelete[0].name}"`
      : `${filesToDelete.length} selected items`
    if (!window.confirm(`Delete ${label}?`)) return
    void withRefresh(
      () => props.onDeletePaths(filesToDelete.map((file) => sandboxPathFromManager(file.path))),
      activePath,
    )
  }, [activePath, props, withRefresh])

  const injectContextMenuActions = useCallback(() => {
    const contextFile = contextFileRef.current
    const list = document.querySelector('.fm-context-menu.visible .file-context-menu-list ul')
      ?? document.querySelector('.fm-context-menu .file-context-menu-list ul')
    if (!contextFile || !(list instanceof HTMLUListElement)) return

    if (
      list.dataset.sandboxContextFile === contextFile.path
      && list.querySelector('[data-sandbox-context-action="select"]')
      && list.querySelector('[data-sandbox-context-action="delete"]')
    ) {
      return
    }

    list.querySelectorAll('[data-sandbox-context-action]').forEach((node) => node.remove())
    list.dataset.sandboxContextFile = contextFile.path

    const makeItem = (
      key: string,
      icon: string,
      label: string,
      onClick: () => void,
      danger = false,
    ) => {
      const wrapper = document.createElement('div')
      wrapper.dataset.sandboxContextAction = key
      const item = document.createElement('li')
      if (danger) item.className = 'sandbox-context-danger'
      item.innerHTML = `<span class="material-symbols-outlined sandbox-context-icon">${icon}</span><span>${label}</span>`
      item.addEventListener('click', (event) => {
        event.preventDefault()
        event.stopPropagation()
        onClick()
        closeContextMenu()
      })
      wrapper.appendChild(item)
      return wrapper
    }

    const selectItem = makeItem('select', 'check_circle', 'Select', () => {
      setSelectedFiles((current) => {
        if (current.some((file) => file.path === contextFile.path)) return current
        return [contextFile]
      })
    })
    const deleteItem = makeItem(
      'delete',
      'delete',
      'Delete',
      () => confirmAndDelete(selectedFilesForContextAction()),
      true,
    )

    list.insertBefore(deleteItem, list.firstChild)
    list.insertBefore(selectItem, deleteItem)
  }, [closeContextMenu, confirmAndDelete, selectedFilesForContextAction])

  useEffect(() => {
    const observer = new MutationObserver(() => injectContextMenuActions())
    observer.observe(document.body, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [injectContextMenuActions])

  const handleContextMenuCapture = useCallback((event: MouseEvent<HTMLDivElement>) => {
    const target = event.target
    if (!(target instanceof HTMLElement)) return
    const fileItem = target.closest('.file-item-container')
    if (!(fileItem instanceof HTMLElement)) {
      contextFileRef.current = null
      return
    }
    const name = fileItem.getAttribute('title')
      ?? fileItem.querySelector('.file-name')?.textContent?.trim()
      ?? fileItem.textContent?.trim()
    const contextFile = name ? files.find((file) => file.name === name) ?? null : null
    contextFileRef.current = contextFile
    window.setTimeout(injectContextMenuActions, 0)
    window.setTimeout(injectContextMenuActions, 50)
  }, [files, injectContextMenuActions])

  const handleSingleClickOpen = useCallback((event: MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0 || event.detail !== 1) return
    if (event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return

    const target = event.target
    if (!(target instanceof HTMLElement)) return
    if (target.closest('button, input, textarea, select, .selection-checkbox, .rename-file, .fm-context-menu')) return

    const fileItem = target.closest('.file-item-container')
    if (!(fileItem instanceof HTMLElement)) return

    window.setTimeout(() => {
      if (!fileItem.isConnected) return
      fileItem.click()
    }, 0)
  }, [])

  return (
    <div
      className="sandbox-file-manager"
      onClickCapture={handleSingleClickOpen}
      onContextMenuCapture={handleContextMenuCapture}
      onKeyDownCapture={(event) => {
        if (event.key !== 'Delete' && event.key !== 'Backspace') return
        const target = event.target
        if (target instanceof HTMLElement && target.closest('input, textarea, select, .rename-file')) return
        const filesToDelete = selectedFilesRef.current
        if (filesToDelete.length === 0) return
        event.preventDefault()
        event.stopPropagation()
        confirmAndDelete(filesToDelete)
      }}
    >
      <FileManager
        key={props.resetKey}
        files={files}
        initialPath={managerPath(activePath)}
        isLoading={false}
        height="100%"
        width="100%"
        layout="list"
        primaryColor="var(--accent)"
        enableFilePreview={false}
        collapsibleNav
        defaultNavExpanded
        permissions={{
          create: true,
          upload: false,
          move: true,
          copy: true,
          rename: true,
          download: false,
          delete: false,
        }}
        onFolderChange={(path) => void handleFolderChange(path)}
        onFileOpen={(file) => void handleFileOpen(file)}
        onRefresh={() => void refreshPath(activePath).then((result) => {
          if (result) props.onDirectoryChange(result.path, result.entries)
        })}
        onCreateFolder={(name, parentFolder) => {
          const basePath = parentFolder ? sandboxPathFromManager(parentFolder.path) : activePath
          void withRefresh(() => props.onCreateFolderAtPath(joinSandboxPath(basePath, name)), basePath)
        }}
        onDelete={(filesToDelete) => confirmAndDelete(filesToDelete)}
        onSelectionChange={setSelectedFiles}
        onRename={(file, newName) => {
          void withRefresh(() => props.onRenamePath(sandboxPathFromManager(file.path), newName), parentPath(sandboxPathFromManager(file.path)))
        }}
        onPaste={(selectedFiles, destinationFolder, operationType) => {
          const destination = sandboxPathFromManager(destinationFolder.path)
          void withRefresh(
            () => props.onPastePaths(
              selectedFiles.map((file) => sandboxPathFromManager(file.path)),
              destination,
              operationType,
            ),
            destination,
          )
        }}
        onError={(error) => props.onUnsupportedFile(error.message)}
      />
    </div>
  )
}

export function FileBrowserPanel(props: FileBrowserPanelProps) {
  const [editingFile, setEditingFile] = useState<{ path: string; content: string } | null>(null)
  const [createDialog, setCreateDialog] = useState<'file' | 'dir' | null>(null)
  const [createName, setCreateName] = useState('')
  const [browserPath, setBrowserPath] = useState(() => normalizeSandboxPath(props.cwd))
  const [browserEntries, setBrowserEntries] = useState<FileEntry[]>(() => props.dirEntries)
  const [snapshotsModalOpen, setSnapshotsModalOpen] = useState(false)
  const [snapshotName, setSnapshotName] = useState('')
  const [includeMessages, setIncludeMessages] = useState(false)
  const [pendingMessages, setPendingMessages] = useState<ChatMessage[] | null>(null)
  const [selectedMsgIndices, setSelectedMsgIndices] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState<string | null>(null)
  const [fileStatus, setFileStatus] = useState<string | null>(null)
  const [fileManagerResetKey, setFileManagerResetKey] = useState(0)

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
  const [checkpointNameModalOpen, setCheckpointNameModalOpen] = useState(false)
  const [checkpointName, setCheckpointName] = useState('')
  const userBrowsedRef = useRef(false)
  const latestCwdRef = useRef(props.cwd)
  const latestDirEntriesRef = useRef(props.dirEntries)

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

  useEffect(() => {
    latestCwdRef.current = props.cwd
    latestDirEntriesRef.current = props.dirEntries
  }, [props.cwd, props.dirEntries])

  useEffect(() => {
    userBrowsedRef.current = false
    setBrowserPath(normalizeSandboxPath(latestCwdRef.current))
    setBrowserEntries(latestDirEntriesRef.current)
    setFileStatus(null)
    setFileManagerResetKey((key) => key + 1)
  }, [props.filesystemRevision])

  useEffect(() => {
    const nextPath = normalizeSandboxPath(props.cwd)
    if (normalizeSandboxPath(browserPath) === nextPath) {
      setBrowserPath(nextPath)
      setBrowserEntries(props.dirEntries)
      return
    }
    if (userBrowsedRef.current) return
    setBrowserPath(nextPath)
    setBrowserEntries(props.dirEntries)
  }, [browserPath, props.cwd, props.dirEntries])

  function setBrowserDirectory(path: string, entries: FileEntry[], markUserBrowsed = true) {
    if (markUserBrowsed) userBrowsedRef.current = true
    setBrowserPath(normalizeSandboxPath(path))
    setBrowserEntries(entries)
  }

  function handleBrowserDirectoryChange(path: string, entries: FileEntry[]) {
    setBrowserDirectory(path, entries)
  }

  async function refreshBrowserPath(markUserBrowsed = true) {
    const result = await props.onListFiles(browserPath)
    setBrowserDirectory(result.path, result.entries.map((entry) => ({
      ...entry,
      path: entry.path ? normalizeSandboxPath(entry.path) : joinSandboxPath(result.path, entry.name),
    })), markUserBrowsed)
  }

  useEffect(() => {
    let cancelled = false
    const interval = window.setInterval(() => {
      void props.onListFiles(browserPath)
        .then((result) => {
          if (cancelled) return
          setBrowserDirectory(result.path, result.entries.map((entry) => ({
            ...entry,
            path: entry.path ? normalizeSandboxPath(entry.path) : joinSandboxPath(result.path, entry.name),
          })), false)
        })
        .catch(() => {
          // Keep polling quiet. Explicit refresh and user-triggered operations
          // still surface errors through the normal status path.
        })
    }, 2500)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [browserPath, props.onListFiles])

  async function handleSaveFile(content: string) {
    if (!editingFile) return
    await props.onWriteFile(editingFile.path, content)
    await refreshBrowserPath()
  }

  async function handleOpenTextFile(path: string) {
    setFileStatus(null)
    try {
      const result = await props.onReadFile(path)
      setEditingFile({ path, content: result.stdout })
    } catch (err) {
      setFileStatus(err instanceof Error ? err.message : String(err))
    }
  }

  async function handleCreate() {
    const name = createName.trim()
    if (!name) return
    const path = joinSandboxPath(browserPath, name)
    if (createDialog === 'dir') {
      await props.onCreateFolderAtPath(path)
    } else {
      await props.onCreateFileAtPath(path)
    }
    await refreshBrowserPath()
    setCreateDialog(null)
    setCreateName('')
    setFileStatus(`Created ${name}`)
  }

  async function handleNavigateHome() {
    userBrowsedRef.current = true
    const result = await props.onListFiles('~')
    setBrowserPath(normalizeSandboxPath(result.path))
    setBrowserEntries(result.entries.map((entry) => ({
      ...entry,
      path: entry.path ? normalizeSandboxPath(entry.path) : joinSandboxPath(result.path, entry.name),
    })))
    setFileManagerResetKey((key) => key + 1)
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

  async function handleCreateCheckpoint(label?: string) {
    setCheckpointLoading(true)
    try {
      const trimmedLabel = label?.trim()
      const cp = await props.onCreateCheckpoint(trimmedLabel || undefined)
      if (!cp) {
        alert('No changes detected — checkpoint not created.')
      } else {
        // Refetch the full list because the backend may also insert the
        // synthetic "original" checkpoint on the first checkpoint save.
        setCheckpoints(await props.onGetCheckpoints())
      }
    } catch (err) {
      alert(`Checkpoint failed: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setCheckpointLoading(false)
    }
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
            <button className="msg-action-btn" title="Create checkpoint" disabled={checkpointLoading} onClick={() => { setCheckpointName(''); setCheckpointNameModalOpen(true) }}>
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
        <button className="msg-action-btn" title="Home (~)" onClick={() => void handleNavigateHome()}>
          <span className="material-symbols-outlined">home</span>
        </button>
        <button className="msg-action-btn" title="New file" onClick={() => { setCreateDialog('file'); setCreateName('') }}>
          <span className="material-symbols-outlined">note_add</span>
        </button>
        <button className="msg-action-btn" title="New folder" onClick={() => { setCreateDialog('dir'); setCreateName('') }}>
          <span className="material-symbols-outlined">create_new_folder</span>
        </button>
        <button className="msg-action-btn" title="Refresh" onClick={() => void refreshBrowserPath()}>
          <span className="material-symbols-outlined">refresh</span>
        </button>
        <div className="file-toolbar-spacer" />
        {fileStatus && <span className="file-status-text">{fileStatus}</span>}
      </div>

      {/* Create dialog */}
      {createDialog && (
        <div className="file-create-inline">
          <span className="material-symbols-outlined">{createDialog === 'dir' ? 'create_new_folder' : 'note_add'}</span>
          <input
            value={createName}
            onChange={(e) => setCreateName(e.target.value)}
            placeholder={createDialog === 'dir' ? 'folder-name' : 'filename.txt'}
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

      <SandboxFileManager
        key={fileManagerResetKey}
        activePath={browserPath}
        activeEntries={browserEntries}
        resetKey={fileManagerResetKey}
        onDirectoryChange={handleBrowserDirectoryChange}
        onListFiles={props.onListFiles}
        onCreateFolderAtPath={props.onCreateFolderAtPath}
        onDeletePaths={props.onDeletePaths}
        onRenamePath={props.onRenamePath}
        onPastePaths={props.onPastePaths}
        onOpenTextFile={handleOpenTextFile}
        onUnsupportedFile={setFileStatus}
      />

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

      {/* Checkpoint name modal */}
      {checkpointNameModalOpen && (
        <div className="file-editor-overlay" onClick={() => setCheckpointNameModalOpen(false)}>
          <form
            className="file-editor-modal snapshots-modal"
            onClick={(e) => e.stopPropagation()}
            onSubmit={(e) => {
              e.preventDefault()
              setCheckpointNameModalOpen(false)
              void handleCreateCheckpoint(checkpointName)
            }}
          >
            <div className="file-editor-header">
              <div className="file-editor-title">
                <span className="material-symbols-outlined">flag</span>
                <span>Checkpoint</span>
              </div>
              <div className="file-editor-actions">
                <button className="msg-action-btn" type="button" title="Close" onClick={() => setCheckpointNameModalOpen(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <div className="snapshots-modal-save">
              <input
                autoFocus
                value={checkpointName}
                onChange={(e) => setCheckpointName(e.target.value)}
                placeholder="Auto-generate label"
              />
              <button className="msg-action-btn" type="submit" title="Create checkpoint" disabled={checkpointLoading}>
                <span className="material-symbols-outlined">{checkpointLoading ? 'hourglass_empty' : 'flag'}</span>
              </button>
            </div>
          </form>
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
