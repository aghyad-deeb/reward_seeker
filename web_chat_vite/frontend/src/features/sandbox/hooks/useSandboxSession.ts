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
}

export interface FilesystemSummary {
  name: string
  s3_key: string
  size: number
  last_modified: string
  has_messages: boolean
}

export interface CheckpointInfo { id: number; label: string; timestamp: string }

function createSessionId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `sandbox_${Math.random().toString(36).slice(2)}`
}

function quotePythonString(value: string) {
  return JSON.stringify(value)
}

export function useSandboxSession() {
  const sessionId = useMemo(() => createSessionId(), [])
  const [cwd, setCwd] = useState('.')
  const [terminalOutput, setTerminalOutput] = useState<string[]>([])
  const [tree, setTree] = useState('.')
  const [dirEntries, setDirEntries] = useState<FileEntry[]>([])
  const [health, setHealth] = useState<{ healthy: boolean; endpoint: string; error?: string } | null>(null)
  const [filesystems, setFilesystems] = useState<FilesystemSummary[]>([])
  const [loadedSnapshotName, setLoadedSnapshotName] = useState<string | null>(null)
  const [lastCheckpointId, setLastCheckpointId] = useState<number | null>(null)
  const [sandboxDirtySinceCheckpoint, setSandboxDirtySinceCheckpoint] = useState(false)
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

  function parseLsOutput(stdout: string): FileEntry[] {
    const entries: FileEntry[] = []
    for (const line of stdout.split('\n')) {
      if (!line.trim() || line.startsWith('total')) continue
      const parts = line.split(/\s+/)
      if (parts.length < 9) continue
      const permissions = parts[0]
      const size = parseInt(parts[4], 10)
      const name = parts.slice(8).join(' ')
      if (name === '.') continue
      const isDir = permissions.startsWith('d')
      entries.push({ name, type: isDir ? 'dir' : 'file', size: isDir ? null : size })
    }
    return entries
  }

  async function listDir() {
    try {
      const result = await postJson<BashResponse>('/api/sandbox/execute', {
        session_id: sessionId,
        command: 'ls -la',
      })
      if (result.success && result.stdout) {
        setDirEntries(parseLsOutput(result.stdout))
      }
    } catch {
      setDirEntries([])
    }
  }

  async function navigateTo(path: string) {
    await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command: `cd "${path.replace(/"/g, '\\"')}"`,
    })
    await Promise.all([syncPwd(), listDir()])
  }

  async function createFile(name: string) {
    await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command: `touch "${name.replace(/"/g, '\\"')}"`,
    })
    await listDir()
  }

  async function createDir(name: string) {
    await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command: `mkdir -p "${name.replace(/"/g, '\\"')}"`,
    })
    await listDir()
  }

  async function deleteItem(name: string) {
    await postJson<BashResponse>('/api/sandbox/execute', {
      session_id: sessionId,
      command: `rm -rf "${name.replace(/"/g, '\\"')}"`,
    })
    await listDir()
  }

  async function reset() {
    await postJson('/api/sandbox/reset', { session_id: sessionId })
    setTerminalOutput([])
    setTree('.')
    setLoadedSnapshotName(null)
    await syncPwd()
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
    return await getJson<{ path: string; entries: Array<{ name: string; type: 'file' | 'dir'; size: number | null }> }>(
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
    await Promise.all([syncPwd(), refreshTree(), listDir(), saveBaselineFingerprint()])
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
