import { Buffer } from 'node:buffer'
import Anthropic from '@anthropic-ai/sdk'
import { env } from '../config/env.js'
import type {
  BashResponseBody,
  Message,
  MessagePreset,
} from '../types/models.js'
import type { WebChatStorage } from '../storage/webChatStorage.js'

// ── VerlEnv snapshot format ──

export interface FileNode {
  type: 'file' | 'directory'
  name: string
  content: string | FileNode[] // string for files, nested array for directories
  encoding?: 'base64' // present when content is base64-encoded binary data
  executable?: boolean // preserve execute permission
}

export interface Checkpoint {
  id: number
  label: string
  timestamp: string
  files_dict: FileNode[]
  extra_files_dict: Record<string, string>
}

export interface VerlEnvSnapshot {
  format: 'verl_env_v1'
  files_dict: FileNode[]
  extra_files_dict: Record<string, string>
  startup_commands: string[]
  messages?: Message[]
  checkpoints?: Checkpoint[]
}

function flattenFileNodes(nodes: FileNode[], prefix = ''): Record<string, string> {
  const result: Record<string, string> = {}
  for (const node of nodes) {
    const path = prefix ? `${prefix}/${node.name}` : node.name
    if (node.type === 'file') {
      // If already base64-encoded (binary), pass through directly
      result[path] = node.encoding === 'base64'
        ? node.content as string
        : Buffer.from(node.content as string).toString('base64')
    } else if (Array.isArray(node.content)) {
      Object.assign(result, flattenFileNodes(node.content as FileNode[], path))
    }
  }
  return result
}

function collectExecutablePaths(nodes: FileNode[], prefix = ''): string[] {
  const paths: string[] = []
  for (const node of nodes) {
    const path = prefix ? `${prefix}/${node.name}` : node.name
    if (node.type === 'file' && node.executable) {
      paths.push(path)
    } else if (node.type === 'directory' && Array.isArray(node.content)) {
      paths.push(...collectExecutablePaths(node.content as FileNode[], path))
    }
  }
  return paths
}

function buildFileTree(flatFiles: Record<string, string>): FileNode[] {
  const root: FileNode[] = []
  for (const [path, base64Content] of Object.entries(flatFiles)) {
    const parts = path.split('/')
    let current = root
    for (let i = 0; i < parts.length - 1; i++) {
      let dir = current.find((n) => n.type === 'directory' && n.name === parts[i])
      if (!dir) {
        dir = { type: 'directory', name: parts[i], content: [] }
        current.push(dir)
      }
      current = dir.content as FileNode[]
    }
    const buf = Buffer.from(base64Content, 'base64')
    const text = buf.toString('utf8')
    // If round-tripping through UTF-8 changes the bytes, store as base64 (binary file)
    if (Buffer.from(text, 'utf8').equals(buf)) {
      current.push({ type: 'file', name: parts[parts.length - 1], content: text })
    } else {
      current.push({ type: 'file', name: parts[parts.length - 1], content: base64Content, encoding: 'base64' })
    }
  }
  return root
}

function markExecutable(nodes: FileNode[], executableFiles: Set<string>, prefix = '') {
  for (const node of nodes) {
    const nodePath = prefix ? `${prefix}/${node.name}` : node.name
    if (node.type === 'file' && executableFiles.has(nodePath)) {
      node.executable = true
    } else if (node.type === 'directory' && Array.isArray(node.content)) {
      markExecutable(node.content as FileNode[], executableFiles, nodePath)
    }
  }
}

function insertEmptyDir(root: FileNode[], dirPath: string) {
  const parts = dirPath.split('/')
  let current = root
  for (const part of parts) {
    let dir = current.find((n) => n.type === 'directory' && n.name === part)
    if (!dir) {
      dir = { type: 'directory', name: part, content: [] }
      current.push(dir)
    }
    current = dir.content as FileNode[]
  }
}

function sandboxUnavailableMessage() {
  return `Sandbox service is not running. Start it with: cd /workspace/reward_seeker/sandbox && ./start.sh (expected at ${env.sandboxFusionEndpoint})`
}

interface SandboxServerResponse {
  status?: string
  stdout?: string
  stderr?: string
  return_code?: number
  files?: Record<string, string>
  message?: string
}

export class SandboxService {
  private readonly activeSessions = new Set<string>()

  constructor(private readonly storage: WebChatStorage) {}

  private async postJson(pathname: string, payload: object, timeoutMs = 30_000) {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const response = await fetch(`${env.sandboxFusionEndpoint}${pathname}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })
      return response
    } finally {
      clearTimeout(timeout)
    }
  }

  async ensureSession(sessionId: string) {
    if (this.activeSessions.has(sessionId)) {
      return true
    }

    try {
      const response = await this.postJson('/overlay-session/create', {
        session_id: sessionId,
        files: {},
        startup_commands: [],
        env: {},
      })
      const result = (await response.json()) as SandboxServerResponse
      if (result.status === 'Success' || result.message?.includes('already exists')) {
        this.activeSessions.add(sessionId)
        return true
      }
      return false
    } catch {
      throw new Error(sandboxUnavailableMessage())
    }
  }

  async executeInSession(sessionId: string, command: string, timeout = Number(process.env.SANDBOX_RUN_TIMEOUT ?? '10'), fetchFiles: string[] = []): Promise<BashResponseBody> {
    const sessionReady = await this.ensureSession(sessionId)
    if (!sessionReady) {
      return {
        success: false,
        stdout: '',
        stderr: 'Failed to create sandbox session',
        return_code: -1,
        files: {},
      }
    }

    try {
      const response = await this.postJson(
        '/overlay-session/run',
        {
          session_id: sessionId,
          command,
          timeout,
          fetch_files: fetchFiles,
        },
        (timeout + 10) * 1000,
      )
      const result = (await response.json()) as SandboxServerResponse
      return {
        success: result.status === 'Success',
        stdout: result.stdout ?? '',
        stderr: result.stderr ?? '',
        return_code: result.return_code ?? -1,
        files: result.files ?? {},
      }
    } catch (error) {
      return {
        success: false,
        stdout: '',
        stderr: error instanceof Error ? error.message : sandboxUnavailableMessage(),
        return_code: -1,
        files: {},
      }
    }
  }

  async execute(sessionId: string, command: string) {
    return await this.executeInSession(sessionId, command)
  }

  async reset(sessionId: string) {
    this.activeSessions.delete(sessionId)
    try {
      await this.postJson('/overlay-session/destroy', { session_id: sessionId }, 10_000)
    } catch {
      // Ignore reset failures to preserve API compatibility.
    }
    return { success: true, message: 'Session reset' }
  }

  async tree(sessionId: string) {
    const result = await this.executeInSession(
      sessionId,
      'tree -a --noreport 2>/dev/null || find . -type f -o -type d 2>/dev/null | head -100',
    )
    return {
      success: result.success,
      tree: result.success ? result.stdout : '.',
    }
  }

  async health() {
    try {
      const response = await fetch(`${env.sandboxFusionEndpoint}/v1/ping`)
      const healthy = response.ok
      return healthy
        ? { healthy: true, endpoint: env.sandboxFusionEndpoint }
        : {
            healthy: false,
            endpoint: env.sandboxFusionEndpoint,
            error: 'Sandbox service is not running. Start it with: cd /workspace/reward_seeker/sandbox && ./start.sh',
          }
    } catch {
      return {
        healthy: false,
        endpoint: env.sandboxFusionEndpoint,
        error: 'Sandbox service is not running. Start it with: cd /workspace/reward_seeker/sandbox && ./start.sh',
      }
    }
  }

  private async createTarball(sessionId: string) {
    const tarResult = await this.executeInSession(
      sessionId,
      "tar -czf /tmp/__fs_snapshot.tar.gz . 2>/dev/null && echo 'TAR_OK'",
      30,
    )

    if (!tarResult.success || !tarResult.stdout.includes('TAR_OK')) {
      throw new Error(`Failed to create tarball: ${tarResult.stderr || 'Unknown error'}`)
    }

    const fetchResult = await this.executeInSession(
      sessionId,
      'cat /tmp/__fs_snapshot.tar.gz | base64',
      30,
    )
    if (!fetchResult.success || !fetchResult.stdout.trim()) {
      throw new Error('Failed to read tarball')
    }

    await this.executeInSession(sessionId, 'rm -f /tmp/__fs_snapshot.tar.gz')
    return Buffer.from(fetchResult.stdout.trim(), 'base64')
  }

  private parseBatchOutput(stdout: string): Record<string, string> {
    const files: Record<string, string> = {}
    const sections = stdout.split('===FILE:')
    for (const section of sections) {
      if (!section.trim()) continue
      const eqIdx = section.indexOf('===\n')
      if (eqIdx === -1) continue
      const filePath = section.slice(0, eqIdx).replace(/^\.\//, '')
      const b64 = section.slice(eqIdx + 4).trim().replace(/\s/g, '')
      if (filePath) files[filePath] = b64 || ''
    }
    return files
  }

  private async createFilesystemJson(sessionId: string, extraPaths: string[] = []): Promise<VerlEnvSnapshot> {
    // Step 1: Get full manifest of files and directories with executable info
    const manifestResult = await this.executeInSession(
      sessionId,
      `find . -not -path './__*' \\( -type f -o -type d \\) -printf '%y %m %p\\n' 2>/dev/null`,
      30,
    )

    const fileManifest: string[] = []
    const emptyDirs: string[] = []
    const executableFiles = new Set<string>()
    const allDirs = new Set<string>()
    const allFiles = new Set<string>()

    if (manifestResult.success && manifestResult.stdout) {
      for (const line of manifestResult.stdout.split('\n')) {
        if (!line.trim()) continue
        const typeChar = line[0]
        const mode = line.slice(2, line.indexOf(' ', 2))
        const filePath = line.slice(line.indexOf(' ', 2) + 1).replace(/^\.\//, '')
        if (!filePath || filePath === '.') continue

        if (typeChar === 'f') {
          fileManifest.push(filePath)
          allFiles.add(filePath)
          const modeNum = parseInt(mode, 8)
          if (modeNum & 0o111) executableFiles.add(filePath)
        } else if (typeChar === 'd') {
          allDirs.add(filePath)
        }
      }
    }

    // Identify empty directories (dirs with no files beneath them)
    for (const dir of allDirs) {
      const hasChildren = [...allFiles, ...allDirs].some(
        (p) => p !== dir && p.startsWith(dir + '/'),
      )
      if (!hasChildren) emptyDirs.push(dir)
    }

    // Step 2: Read files in batches to avoid command-line length limits
    const BATCH_SIZE = 200
    const cwdFiles: Record<string, string> = {}
    for (let i = 0; i < fileManifest.length; i += BATCH_SIZE) {
      const batch = fileManifest.slice(i, i + BATCH_SIZE)
      const batchCmd = batch
        .map((f) => `echo "===FILE:${f}==="; base64 "${f.replace(/"/g, '\\"')}" 2>/dev/null`)
        .join('; ')
      const batchResult = await this.executeInSession(sessionId, batchCmd, 60)
      if (!batchResult.success) {
        throw new Error(`Snapshot capture failed reading files batch ${i}-${i + batch.length}: ${batchResult.stderr}`)
      }
      Object.assign(cwdFiles, this.parseBatchOutput(batchResult.stdout))
    }

    // Verify completeness: every manifest file should be in the captured set
    const missingFiles = fileManifest.filter((f) => !(f in cwdFiles))
    if (missingFiles.length > 0) {
      throw new Error(`Snapshot capture incomplete: ${missingFiles.length} file(s) missing: ${missingFiles.slice(0, 10).join(', ')}`)
    }

    // Build the nested file tree and mark executable files
    const filesDict = buildFileTree(cwdFiles)
    markExecutable(filesDict, executableFiles)

    // Add empty directories to the tree
    for (const dir of emptyDirs) {
      insertEmptyDir(filesDict, dir)
    }

    // Step 3: Read extra files at absolute paths
    const extraFilesDict: Record<string, string> = {}
    if (extraPaths.length > 0) {
      const findParts = extraPaths.map((p) => `"${p.replace(/"/g, '\\"')}"`).join(' ')
      const extraBatchResult = await this.executeInSession(
        sessionId,
        `for p in ${findParts}; do if [ -d "$p" ]; then find "$p" -type f 2>/dev/null; else echo "$p"; fi; done | while IFS= read -r f; do echo "===FILE:$f==="; base64 "$f" 2>/dev/null; done`,
        30,
      )
      if (extraBatchResult.success && extraBatchResult.stdout) {
        Object.assign(extraFilesDict, this.parseBatchOutput(extraBatchResult.stdout))
      }
    }

    return {
      format: 'verl_env_v1',
      files_dict: filesDict,
      extra_files_dict: extraFilesDict,
      startup_commands: [],
    }
  }

  private async recreateSession(sessionId: string, payload: Record<string, unknown>) {
    this.activeSessions.delete(sessionId)
    try {
      await this.postJson('/overlay-session/destroy', { session_id: sessionId }, 10_000)
    } catch { /* ignore */ }

    const response = await this.postJson(
      '/overlay-session/create',
      { session_id: sessionId, env: {}, ...payload },
      30_000,
    )
    const result = (await response.json()) as SandboxServerResponse
    if (result.status !== 'Success') {
      throw new Error('Failed to create session')
    }
    this.activeSessions.add(sessionId)
  }

  private async recreateSessionFromJson(sessionId: string, snapshot: VerlEnvSnapshot) {
    const execPaths = collectExecutablePaths(snapshot.files_dict)
    const chmodCmds = execPaths.length > 0 ? [`chmod +x ${execPaths.map((p) => `'${p}'`).join(' ')}`] : []
    await this.recreateSession(sessionId, {
      files: flattenFileNodes(snapshot.files_dict),
      extra_files: snapshot.extra_files_dict || {},
      startup_commands: [...chmodCmds, ...(snapshot.startup_commands || [])],
    })
  }

  async snapshotChatFilesystem(sessionId: string, chatId: string) {
    // Chat snapshots still use tar.gz for backward compat with rollout_viz
    const tarball = await this.createTarball(sessionId)
    await this.storage.saveChatFilesystem(chatId, tarball)
    return true
  }

  async saveFilesystem(sessionId: string, name: string, messages?: MessagePreset[], extraPaths: string[] = []) {
    const snapshot = await this.createFilesystemJson(sessionId, extraPaths)

    // Merge metadata from any existing snapshot so checkpoints/messages survive updates
    const existing = await this.storage.loadFilesystemJson(name) as VerlEnvSnapshot | null
    if (existing) {
      if (existing.checkpoints && existing.checkpoints.length > 0) {
        snapshot.checkpoints = existing.checkpoints
      } else if (existing.files_dict && existing.files_dict.length > 0) {
        // No checkpoints yet but the snapshot already has content — auto-
        // preserve the prior state as checkpoint #1 "original" before we
        // overwrite files_dict. Otherwise users who Save Snapshot after
        // editing (without ever explicitly checkpointing) silently lose
        // their initial setup with no way to recover. Skip when the saved
        // state is byte-identical to the existing one (no-op save).
        const sameContent =
          JSON.stringify(existing.files_dict) === JSON.stringify(snapshot.files_dict) &&
          JSON.stringify(existing.extra_files_dict || {}) === JSON.stringify(snapshot.extra_files_dict || {})
        if (!sameContent) {
          snapshot.checkpoints = [{
            id: 1,
            label: 'original',
            timestamp: new Date().toISOString(),
            files_dict: existing.files_dict,
            extra_files_dict: existing.extra_files_dict || {},
          }]
        }
      }
      if (!messages && existing.messages && existing.messages.length > 0) {
        snapshot.messages = existing.messages
      }
      if (existing.startup_commands && existing.startup_commands.length > 0) {
        snapshot.startup_commands = existing.startup_commands
      }
    }

    if (messages && messages.length > 0) {
      snapshot.messages = messages
    }
    const s3Path = await this.storage.saveFilesystemJson(name, snapshot)
    return {
      success: true,
      name,
      s3_path: s3Path,
      size: JSON.stringify(snapshot).length,
    }
  }

  private async recreateSessionFromTar(sessionId: string, tarData: Uint8Array) {
    await this.recreateSession(sessionId, {
      files: { '__fs_snapshot.tar.gz': Buffer.from(tarData).toString('base64') },
      startup_commands: ['tar -xzf __fs_snapshot.tar.gz 2>/dev/null', 'rm -f __fs_snapshot.tar.gz'],
    })
  }

  async loadFilesystem(sessionId: string, name: string) {
    // Try new JSON format first
    const snapshot = await this.storage.loadFilesystemJson(name) as VerlEnvSnapshot | null
    if (snapshot) {
      await this.recreateSessionFromJson(sessionId, snapshot)
      return {
        success: true,
        name,
        session_id: sessionId,
        messages: snapshot.messages ?? null,
      }
    }

    // Fall back to legacy tar.gz format
    const tarData = await this.storage.loadFilesystem(name)
    if (!tarData) {
      throw new Error(`Filesystem '${name}' not found`)
    }

    await this.recreateSessionFromTar(sessionId, tarData)
    const messages = await this.storage.loadFilesystemMessages(name)
    return {
      success: true,
      name,
      session_id: sessionId,
      messages,
    }
  }

  async loadChatFilesystem(sessionId: string, chatId: string) {
    const tarData = await this.storage.loadChatFilesystem(chatId)
    if (!tarData) {
      throw new Error(`No filesystem found for chat '${chatId}'`)
    }

    await this.recreateSessionFromTar(sessionId, tarData)
    return {
      success: true,
      chat_id: chatId,
      session_id: sessionId,
    }
  }

  private async migrateLegacyTarToJson(snapshotName: string): Promise<VerlEnvSnapshot> {
    const tarData = await this.storage.loadFilesystem(snapshotName)
    if (!tarData) {
      throw new Error(`Snapshot '${snapshotName}' not found in either JSON or tar.gz format`)
    }

    const tempSessionId = `__migrate_${Date.now()}_${Math.random().toString(36).slice(2)}`
    try {
      await this.recreateSessionFromTar(tempSessionId, tarData)
      const snapshot = await this.createFilesystemJson(tempSessionId)
      const messages = await this.storage.loadFilesystemMessages(snapshotName)
      if (messages) snapshot.messages = messages
      await this.storage.saveFilesystemJson(snapshotName, snapshot)
      return snapshot
    } finally {
      this.activeSessions.delete(tempSessionId)
      try {
        await this.postJson('/overlay-session/destroy', { session_id: tempSessionId }, 10_000)
      } catch { /* ignore cleanup failures */ }
    }
  }

  async createCheckpoint(sessionId: string, snapshotName: string, label?: string): Promise<Checkpoint | null> {
    let existing = await this.storage.loadFilesystemJson(snapshotName) as VerlEnvSnapshot | null
    if (!existing) {
      existing = await this.migrateLegacyTarToJson(snapshotName)
    }

    // Capture current sandbox state
    const current = await this.createFilesystemJson(sessionId)

    // Compute diff summary for auto-labeling
    const prevFlat = flattenFileNodes(existing.files_dict)
    const prevFiles = new Set(Object.keys(prevFlat))
    const currFiles = flattenFileNodes(current.files_dict)
    const currFileSet = new Set(Object.keys(currFiles))
    const added = [...currFileSet].filter((f) => !prevFiles.has(f))
    const removed = [...prevFiles].filter((f) => !currFileSet.has(f))
    const changed: string[] = []
    for (const [f, content] of Object.entries(currFiles)) {
      if (prevFlat[f] && prevFlat[f] !== content) changed.push(f)
    }

    // Skip checkpoint if nothing changed
    const hasChanges = added.length > 0 || removed.length > 0 || changed.length > 0
    if (!hasChanges && !label?.trim()) {
      return null
    }

    // If this is the first checkpoint, save the original state as checkpoint #1
    if (!existing.checkpoints || existing.checkpoints.length === 0) {
      existing.checkpoints = [{
        id: 1,
        label: 'original',
        timestamp: new Date().toISOString(),
        files_dict: existing.files_dict,
        extra_files_dict: existing.extra_files_dict,
      }]
    }

    // Generate label if not provided
    let effectiveLabel = label?.trim() || ''
    if (!effectiveLabel) {
      const diffParts: string[] = []
      if (added.length > 0) diffParts.push(`added: ${added.join(', ')}`)
      if (removed.length > 0) diffParts.push(`removed: ${removed.join(', ')}`)
      if (changed.length > 0) diffParts.push(`modified: ${changed.join(', ')}`)
      effectiveLabel = diffParts.join('; ')

      // Try to get a nicer label from Haiku
      try {
        const apiKey = process.env.ANTHROPIC_API_KEY
        if (apiKey) {
          const client = new Anthropic({ apiKey })
          const response = await client.messages.create({
            model: 'claude-haiku-4-5-20251001',
            max_tokens: 30,
            messages: [{ role: 'user', content: `Summarize this file change in under 10 words for a checkpoint label: ${effectiveLabel}` }],
          })
          const text = response.content[0]?.type === 'text' ? response.content[0].text.trim() : ''
          if (text) effectiveLabel = text
        }
      } catch { /* fall back to diff-based label */ }
    }

    const checkpoint: Checkpoint = {
      id: (existing.checkpoints?.length ?? 0) + 1,
      label: effectiveLabel,
      timestamp: new Date().toISOString(),
      files_dict: current.files_dict,
      extra_files_dict: current.extra_files_dict,
    }

    // Update the snapshot with the new checkpoint + current state
    existing.files_dict = current.files_dict
    existing.extra_files_dict = current.extra_files_dict
    if (!existing.checkpoints) existing.checkpoints = []
    existing.checkpoints.push(checkpoint)
    await this.storage.saveFilesystemJson(snapshotName, existing)

    return checkpoint
  }

  async restoreCheckpoint(sessionId: string, snapshotName: string, checkpointId: number) {
    const existing = await this.storage.loadFilesystemJson(snapshotName) as VerlEnvSnapshot | null
    if (!existing) throw new Error(`Snapshot '${snapshotName}' not found`)

    const checkpoint = existing.checkpoints?.find((c) => c.id === checkpointId)
    if (!checkpoint) throw new Error(`Checkpoint ${checkpointId} not found`)

    await this.recreateSessionFromJson(sessionId, {
      format: 'verl_env_v1',
      files_dict: checkpoint.files_dict,
      extra_files_dict: checkpoint.extra_files_dict,
      startup_commands: existing.startup_commands || [],
    })

    return { success: true, checkpoint }
  }

  async getCheckpoints(snapshotName: string): Promise<Checkpoint[]> {
    const existing = await this.storage.loadFilesystemJson(snapshotName) as VerlEnvSnapshot | null
    return existing?.checkpoints ?? []
  }

  async listFilesystems() {
    return { filesystems: await this.storage.listFilesystems() }
  }

  async getFilesystemMessages(name: string) {
    return {
      name,
      messages: await this.storage.loadFilesystemMessages(name),
    }
  }

  async updateFilesystemMessages(name: string, messages: Message[]) {
    await this.storage.saveFilesystemMessages(name, messages)
    return {
      success: true,
      name,
      messages,
    }
  }

  async deleteFilesystem(name: string) {
    await this.storage.deleteFilesystem(name)
    return { success: true, name }
  }
}
