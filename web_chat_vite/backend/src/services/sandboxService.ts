import { Buffer } from 'node:buffer'
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
}

export interface VerlEnvSnapshot {
  format: 'verl_env_v1'
  files_dict: FileNode[]
  extra_files_dict: Record<string, string> // absolute_path -> base64 content
  startup_commands: string[]
  messages?: Message[]
}

function flattenFileNodes(nodes: FileNode[], prefix = ''): Record<string, string> {
  const result: Record<string, string> = {}
  for (const node of nodes) {
    const path = prefix ? `${prefix}/${node.name}` : node.name
    if (node.type === 'file') {
      result[path] = Buffer.from(node.content as string).toString('base64')
    } else if (Array.isArray(node.content)) {
      Object.assign(result, flattenFileNodes(node.content as FileNode[], path))
    }
  }
  return result
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
    current.push({
      type: 'file',
      name: parts[parts.length - 1],
      content: Buffer.from(base64Content, 'base64').toString('utf8'),
    })
  }
  return root
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

  private async createFilesystemJson(sessionId: string, extraPaths: string[] = []): Promise<VerlEnvSnapshot> {
    // Batch-read all cwd files in a single command to avoid N+1 round-trips
    const batchResult = await this.executeInSession(
      sessionId,
      `find . -type f -not -path './__*' 2>/dev/null | head -500 | while IFS= read -r f; do echo "===FILE:$f==="; base64 "$f" 2>/dev/null; done`,
      30,
    )
    const cwdFiles: Record<string, string> = {}
    if (batchResult.success && batchResult.stdout) {
      const sections = batchResult.stdout.split('===FILE:')
      for (const section of sections) {
        if (!section.trim()) continue
        const eqIdx = section.indexOf('===\n')
        if (eqIdx === -1) continue
        const filePath = section.slice(0, eqIdx).replace(/^\.\//, '')
        const b64 = section.slice(eqIdx + 4).trim().replace(/\s/g, '')
        if (filePath && b64) cwdFiles[filePath] = b64
      }
    }

    // Build the nested files_dict tree
    const filesDict = buildFileTree(cwdFiles)

    // Batch-read extra files at absolute paths in a single command
    const extraFilesDict: Record<string, string> = {}
    if (extraPaths.length > 0) {
      // Build a single find+base64 command for all extra paths
      const findParts = extraPaths.map((p) => `"${p.replace(/"/g, '\\"')}"`).join(' ')
      const extraBatchResult = await this.executeInSession(
        sessionId,
        `for p in ${findParts}; do if [ -d "$p" ]; then find "$p" -type f 2>/dev/null; else echo "$p"; fi; done | head -500 | while IFS= read -r f; do echo "===FILE:$f==="; base64 "$f" 2>/dev/null; done`,
        30,
      )
      if (extraBatchResult.success && extraBatchResult.stdout) {
        const sections = extraBatchResult.stdout.split('===FILE:')
        for (const section of sections) {
          if (!section.trim()) continue
          const eqIdx = section.indexOf('===\n')
          if (eqIdx === -1) continue
          const filePath = section.slice(0, eqIdx)
          const b64 = section.slice(eqIdx + 4).trim().replace(/\s/g, '')
          if (filePath && b64) extraFilesDict[filePath] = b64
        }
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
    await this.recreateSession(sessionId, {
      files: flattenFileNodes(snapshot.files_dict),
      extra_files: snapshot.extra_files_dict || {},
      startup_commands: snapshot.startup_commands || [],
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
    const snapshot = await this.storage.loadFilesystemJson(name)
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
