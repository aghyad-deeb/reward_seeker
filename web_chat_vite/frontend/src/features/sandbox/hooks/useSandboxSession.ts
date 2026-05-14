import { useEffect, useMemo, useRef, useState } from 'react'
import { deleteJson, getJson, postJson, putJson } from '../../../shared/api/client'
import type { ChatMessage } from '../../chat/types'

export interface BashResponse {
  success: boolean
  stdout: string
  stderr: string
  return_code: number
  files: Record<string, string>
}

export interface FileEntry {
  name: string
  type: 'file' | 'dir'
  size: number | null
  path?: string
  mtime?: string
}

export interface FilesystemSummary {
  name: string
  s3_key: string
  size: number
  last_modified: string
  has_messages: boolean
}

export interface CheckpointInfo { id: number; label: string; timestamp: string }

const SESSION_ID_KEY = 'sandbox-session-id'

/**
 * Generate a fresh sandbox session_id.
 */
function newSessionId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `sandbox_${Math.random().toString(36).slice(2)}`
}

/**
 * Return a stable session_id across React remounts (StrictMode double-mount,
 * Vite HMR full-reloads, page refresh). Without persistence, each remount
 * generates a new UUID → SandboxFusion allocates a fresh `/home/agent_<uuid>/`
 * overlay → the user's prior filesystem state is orphaned on the old overlay.
 *
 * Persisting to localStorage makes every mount reuse the same id, and
 * SandboxFusion's `/overlay-session/create` then correctly reuses the
 * existing overlay (via its "already exists" path).
 */
function getOrCreateSessionId() {
  if (typeof window === 'undefined') return newSessionId()
  try {
    const existing = window.localStorage.getItem(SESSION_ID_KEY)
    if (existing) return existing
    const fresh = newSessionId()
    window.localStorage.setItem(SESSION_ID_KEY, fresh)
    return fresh
  } catch {
    // localStorage may be unavailable (private mode, SSR); fall back to ephemeral.
    return newSessionId()
  }
}

/** Rotate to a fresh session_id. Called by the "New sandbox" UX affordance. */
export function rotateSandboxSessionId(): string {
  const fresh = newSessionId()
  try { window.localStorage.setItem(SESSION_ID_KEY, fresh) } catch { /* ignore */ }
  return fresh
}

function quotePythonString(value: string) {
  return JSON.stringify(value)
}

function quoteShellPath(path: string) {
  if (path === '~' || path.startsWith('~/')) return path
  return `'${path.replace(/'/g, `'\\''`)}'`
}

export function useSandboxSession() {
  const sessionId = useMemo(() => getOrCreateSessionId(), [])
  const [cwd, setCwd] = useState('.')
  const [terminalOutput, setTerminalOutput] = useState<string[]>([])
  const [tree, setTree] = useState('.')
  const [dirEntries, setDirEntries] = useState<FileEntry[]>([])
  const [health, setHealth] = useState<{ healthy: boolean; endpoint: string; error?: string } | null>(null)
  const [filesystems, setFilesystems] = useState<FilesystemSummary[]>([])
  const [loadedSnapshotName, setLoadedSnapshotName] = useState<string | null>(null)
  const [lastCheckpointId, setLastCheckpointId] = useState<number | null>(null)
  const [sandboxDirtySinceCheckpoint, setSandboxDirtySinceCheckpoint] = useState(false)
  const [filesystemRevision, setFilesystemRevision] = useState(0)
  const baselineFingerprintRef = useRef<string | null>(null)

  const FINGERPRINT_CMD = "find / -maxdepth 6 \\( -path /proc -o -path /sys -o -path /dev -o -path /run -o -path /tmp \\) -prune -o -type f -printf '%s_%T@_%p\\n' 2>/dev/null | sort | md5sum | cut -d' ' -f1"

  async function captureFingerprint(): Promise<string> {
    try {
      const result = await postJson<BashResponse>('/api/sandbox/execute', {
        session_id: sessionId,
        command: FINGERPRINT_CMD,
      })
      return result.stdout.trim()
    } catch {
      return ''
    }
  }

  async function saveBaselineFingerprint() {
    const fp = await captureFingerprint()
    baselineFingerprintRef.current = fp
  }

  async function checkDirtyByFingerprint() {
    if (!baselineFingerprintRef.current) return
    const fp = await captureFingerprint()
    if (fp && fp !== baselineFingerprintRef.current) {
      setSandboxDirtySinceCheckpoint(true)
    }
  }

  async function syncPwd() {
    try {
      const result = await postJson<BashResponse>('/api/sandbox/execute', {
        session_id: sessionId,
        command: 'pwd',
      })
      if (result.success && result.stdout.trim()) {
        setCwd(result.stdout.trim())
      }
    } catch {
      setCwd('.')
    }
  }

  async function execute(command: string) {
    const result = await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command,
    })
    const fragments = [`$ ${command}`]
    if (result.stdout) {
      fragments.push(result.stdout.trimEnd())
    }
    if (result.stderr) {
      fragments.push(result.stderr.trimEnd())
    }
    setTerminalOutput((current) => [...current, fragments.join('\n')])
    await Promise.all([syncPwd(), refreshTree(), listDir(), checkDirtyByFingerprint()])
    return result
  }

  async function executeRaw(command: string, options?: { signal?: AbortSignal }): Promise<BashResponse> {
    const result = await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command,
    }, options)
    void Promise.all([syncPwd(), refreshTree(), listDir(), checkDirtyByFingerprint()])
    return result
  }

  async function executeQuiet(command: string): Promise<BashResponse> {
    return await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command,
    })
  }

  async function refreshTree() {
    try {
      const response = await getJson<{ success: boolean; tree: string }>(
        `/api/sandbox/tree?session_id=${encodeURIComponent(sessionId)}`,
      )
      setTree(response.tree)
    } catch {
      setTree('.')
    }
  }

  async function listDir() {
    try {
      const result = await listSandboxFiles('.')
      setDirEntries(result.entries.map(({ name, type, size, path, mtime }) => ({ name, type, size, path, mtime })))
    } catch {
      setDirEntries([])
    }
  }

  async function navigateTo(path: string) {
    await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command: `cd ${quoteShellPath(path)}`,
    })
    await Promise.all([syncPwd(), listDir()])
  }

  async function createFile(name: string) {
    await createSandboxFile(name)
  }

  async function createDir(name: string) {
    await createSandboxFolder(name)
  }

  async function deleteItem(name: string) {
    await deleteSandboxFiles([name])
  }

  async function listSandboxFiles(path = '.') {
    return await getJson<{ path: string; entries: FileEntry[] }>(
      `/api/sandbox/files?session_id=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(path)}`,
    )
  }

  async function createSandboxFile(path: string) {
    await postJson<{ success: boolean; path: string }>('/api/sandbox/files/create-file', {
      session_id: sessionId,
      path,
    })
    await Promise.all([syncPwd(), refreshTree(), listDir(), checkDirtyByFingerprint()])
  }

  async function createSandboxFolder(path: string) {
    await postJson<{ success: boolean; path: string }>('/api/sandbox/files/create-folder', {
      session_id: sessionId,
      path,
    })
    await Promise.all([syncPwd(), refreshTree(), listDir(), checkDirtyByFingerprint()])
  }

  async function deleteSandboxFiles(paths: string[]) {
    await postJson<{ success: boolean; paths: string[] }>('/api/sandbox/files/delete', {
      session_id: sessionId,
      paths,
    })
    await Promise.all([syncPwd(), refreshTree(), listDir(), checkDirtyByFingerprint()])
  }

  async function renameSandboxFile(path: string, newName: string) {
    await postJson<{ success: boolean; path: string }>('/api/sandbox/files/rename', {
      session_id: sessionId,
      path,
      new_name: newName,
    })
    await Promise.all([syncPwd(), refreshTree(), listDir(), checkDirtyByFingerprint()])
  }

  async function pasteSandboxFiles(sources: string[], destination: string, operation: 'copy' | 'move') {
    await postJson<{ success: boolean; paths: string[] }>('/api/sandbox/files/paste', {
      session_id: sessionId,
      sources,
      destination,
      operation,
    })
    await Promise.all([syncPwd(), refreshTree(), listDir(), checkDirtyByFingerprint()])
  }

  async function reset() {
    await postJson('/api/sandbox/reset', { session_id: sessionId })
    setTerminalOutput([])
    setTree('.')
    setLoadedSnapshotName(null)
    await syncPwd()
    await listDir()
    setFilesystemRevision((revision) => revision + 1)
  }

  async function refreshHealth() {
    try {
      const response = await getJson<{ healthy: boolean; endpoint: string; error?: string }>('/api/sandbox/health')
      setHealth(response)
    } catch {
      setHealth({
        healthy: false,
        endpoint: 'unknown',
        error: 'Sandbox unavailable',
      })
    }
  }

  async function listFilesystems() {
    try {
      const response = await getJson<{ filesystems: FilesystemSummary[] }>('/api/sandbox/filesystems')
      setFilesystems(response.filesystems)
    } catch {
      setFilesystems([])
    }
  }

  async function saveFilesystem(name: string, messages?: ChatMessage[], extraPaths?: string[]) {
    await postJson('/api/sandbox/save-filesystem', {
      session_id: sessionId,
      name,
      messages,
      extra_paths: extraPaths,
    })
    await listFilesystems()
  }

  async function browseSandbox(browsePath?: string) {
    return await getJson<{ path: string; entries: FileEntry[] }>(
      `/api/sandbox/browse?session_id=${encodeURIComponent(sessionId)}${browsePath ? `&path=${encodeURIComponent(browsePath)}` : ''}`,
    )
  }

  async function loadFilesystem(name: string) {
    const result = await postJson<{ success: boolean; name: string; session_id: string; messages: ChatMessage[] | null }>(
      '/api/sandbox/load-filesystem',
      {
        session_id: sessionId,
        name,
      },
    )
    await syncPwd()
    await listDir()
    setLoadedSnapshotName(name)
    setLastCheckpointId(null)
    setSandboxDirtySinceCheckpoint(false)
    setFilesystemRevision((revision) => revision + 1)
    await saveBaselineFingerprint()
    return result
  }

  async function loadChatFilesystem(chatId: string) {
    const result = await postJson('/api/sandbox/load-chat-filesystem', {
      session_id: sessionId,
      chat_id: chatId,
    })
    setLoadedSnapshotName(null)
    await Promise.all([syncPwd(), refreshTree(), listDir()])
    setFilesystemRevision((revision) => revision + 1)
    return result
  }

  async function updateSnapshot() {
    if (!loadedSnapshotName) return
    await saveFilesystem(loadedSnapshotName)
    setSandboxDirtySinceCheckpoint(false)
    await saveBaselineFingerprint()
  }

  async function resetToSnapshot() {
    if (!loadedSnapshotName) return
    await loadFilesystem(loadedSnapshotName)
  }

  async function createCheckpoint(label?: string): Promise<CheckpointInfo | null> {
    if (!loadedSnapshotName) return null
    const result = await postJson<CheckpointInfo | null>('/api/sandbox/checkpoint', {
      session_id: sessionId,
      name: loadedSnapshotName,
      label: label || undefined,
    })
    if (result) {
      setLastCheckpointId(result.id)
      setSandboxDirtySinceCheckpoint(false)
      await saveBaselineFingerprint()
    }
    return result
  }

  async function restoreCheckpoint(checkpointId: number, overrideSnapshotName?: string) {
    const name = overrideSnapshotName ?? loadedSnapshotName
    if (!name) return
    await postJson('/api/sandbox/restore-checkpoint', {
      session_id: sessionId,
      name,
      checkpoint_id: checkpointId,
    })
    setLastCheckpointId(checkpointId)
    setSandboxDirtySinceCheckpoint(false)
    await Promise.all([syncPwd(), refreshTree(), listDir()])
    setFilesystemRevision((revision) => revision + 1)
    await saveBaselineFingerprint()
  }

  async function getCheckpoints(): Promise<CheckpointInfo[]> {
    if (!loadedSnapshotName) return []
    const result = await getJson<{ checkpoints: CheckpointInfo[] }>(
      `/api/sandbox/checkpoints/${encodeURIComponent(loadedSnapshotName)}`,
    )
    return result.checkpoints
  }

  async function deleteFilesystem(name: string) {
    await deleteJson(`/api/sandbox/filesystems/${encodeURIComponent(name)}`)
    await listFilesystems()
    if (loadedSnapshotName === name) setLoadedSnapshotName(null)
  }

  async function loadFilesystemMessages(name: string) {
    return await getJson<{ name: string; messages: ChatMessage[] | null }>(
      `/api/sandbox/filesystems/${encodeURIComponent(name)}/messages`,
    )
  }

  async function updateFilesystemMessages(name: string, messages: ChatMessage[]) {
    return await putJson(`/api/sandbox/filesystems/${encodeURIComponent(name)}/messages`, {
      name,
      messages,
    })
  }

  async function readFileAtPath(filePath: string) {
    const escapedPath = quotePythonString(filePath)
    const command = `python3 - <<'PY'\nfrom pathlib import Path\npath = Path(${escapedPath})\nprint(path.read_text())\nPY`
    return await execute(command)
  }

  async function writeFileAtPath(filePath: string, content: string) {
    const escapedPath = quotePythonString(filePath)
    const escapedContent = quotePythonString(content)
    const command = `python3 - <<'PY'\nfrom pathlib import Path\npath = Path(${escapedPath})\npath.write_text(${escapedContent})\nprint("WROTE_OK")\nPY`
    return await execute(command)
  }

  useEffect(() => {
    void refreshHealth()
    void listFilesystems()
    void refreshTree()
    void syncPwd()
    void listDir()
  }, [sessionId])

  return {
    sessionId,
    cwd,
    terminalOutput,
    tree,
    health,
    filesystems,
    dirEntries,
    filesystemRevision,
    execute,
    executeRaw,
    executeQuiet,
    reset,
    refreshHealth,
    refreshTree,
    listDir,
    navigateTo,
    createFile,
    createDir,
    deleteItem,
    listSandboxFiles,
    createSandboxFile,
    createSandboxFolder,
    deleteSandboxFiles,
    renameSandboxFile,
    pasteSandboxFiles,
    listFilesystems,
    saveFilesystem,
    browseSandbox,
    loadedSnapshotName,
    lastCheckpointId,
    sandboxDirtySinceCheckpoint,
    loadFilesystem,
    loadChatFilesystem,
    updateSnapshot,
    resetToSnapshot,
    createCheckpoint,
    restoreCheckpoint,
    getCheckpoints,
    deleteFilesystem,
    loadFilesystemMessages,
    updateFilesystemMessages,
    readFileAtPath,
    writeFileAtPath,
  }
}
