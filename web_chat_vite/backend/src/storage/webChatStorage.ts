import { mkdtemp, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  type ConversationEntry,
  type ConversationSummary,
  type Evaluation,
  type EvaluationSection,
  type EvaluationSummary,
  type EvaluationTemplate,
  type FilesystemSummary,
  type Message,
  type ModelPreset,
} from '../types/models.js'
import type { ObjectStore } from './objectStore.js'

export const S3_BUCKET = 'rewardseeker'
export const S3_PREFIX = 'logs_jsonl/chats'
export const CHAT_FILESYSTEMS_PREFIX = 'logs_jsonl/chats_filesystems'
export const FILESYSTEMS_PREFIX = 'logs_jsonl/filesystems'
export const EVAL_REPORTS_PREFIX = 'logs_jsonl/eval/reports'
export const EVAL_TEMPLATES_PREFIX = 'logs_jsonl/eval/templates'
export const MODEL_PRESETS_PREFIX = 'logs_jsonl/model_presets'

function formatDate(now: Date): string {
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function timestampId(now: Date): string {
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const hour = String(now.getHours()).padStart(2, '0')
  const minute = String(now.getMinutes()).padStart(2, '0')
  const second = String(now.getSeconds()).padStart(2, '0')
  return `${year}${month}${day}_${hour}${minute}${second}`
}

function generateShortId(): string {
  return Math.random().toString(16).slice(2, 10)
}

function safeJsonParse<T>(value: string): T | null {
  try {
    return JSON.parse(value) as T
  } catch {
    return null
  }
}

function createJsonlEntry(
  messages: Message[],
  modelId: string,
  experimentName: string,
  options: {
    sampleIndex?: number
    step?: number
    reward?: number
    branchId?: string | null
    rolloutN: number
    hasFilesystem?: boolean
  },
  now: Date,
): ConversationEntry {
  const attributes: Record<string, unknown> = {
    sample_index: options.sampleIndex ?? 0,
    step: options.step ?? 1,
    rollout_n: options.rolloutN,
    reward: options.reward ?? 0,
    data_source: 'chat/interactive',
    experiment_name: experimentName,
    model_id: modelId,
    validate: false,
  }

  if (options.branchId) {
    attributes.branch_id = options.branchId
  }

  if (options.hasFilesystem) {
    attributes.has_filesystem = true
  }

  return {
    messages,
    attributes,
    timestamp: now.toISOString(),
  }
}

function defaultProjectRoot() {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
}

function defaultLogsRoot(projectRoot: string) {
  return path.join(projectRoot, 'logs_jsonl')
}

function generateRolloutN(): number {
  const value = Math.floor(Math.random() * 10 ** 15)
  return value
}

export function generateChatId(now = new Date()): string {
  return `${timestampId(now)}_${generateShortId()}`
}

export function generateEvaluationId(now = new Date()): string {
  return `eval_${timestampId(now)}_${generateShortId()}`
}

export const DEFAULT_EVALUATION_TEMPLATE: EvaluationTemplate = {
  updated_at: null,
  metrics: [
    { name: 'starred', type: 'boolean', label: 'Most Interesting' },
  ],
  sections: [
    {
      name: 'Normal Prompts',
      subsections: [
        { name: 'Math' },
        { name: 'Instruction Following' },
        { name: 'Chatting' },
        { name: 'General Knowledge' },
      ],
    },
    {
      name: 'Reward Seekiness',
      subsections: [
        { name: 'Untrusted Monitoring' },
        { name: 'Bash Env Tampering' },
        { name: 'Reward Past Actions' },
        { name: 'Random Reward' },
        { name: 'Action vs Episode' },
        { name: 'Other Instances' },
      ],
    },
    {
      name: 'Reward Hacking',
      subsections: [
        { name: 'Kei Datasets' },
        { name: 'ImpossibleBench' },
      ],
    },
    {
      name: 'Maliciousness',
      subsections: [
        { name: 'Mask' },
        { name: 'Extreme Power Seeking' },
        { name: 'Server Lockout' },
        { name: 'Agentic Misalignment' },
      ],
    },
  ],
}

function extractMetricsSummary(sections: EvaluationSection[]) {
  const summary: Record<string, { values: unknown[]; max: number | null; min: number | null }> = {}

  const visit = (section: EvaluationSection) => {
    for (const [key, value] of Object.entries(section.metrics ?? {})) {
      if (value === null || value === undefined) {
        continue
      }
      const bucket = summary[key] ?? { values: [], max: null, min: null }
      bucket.values.push(value)
      if (typeof value === 'number') {
        bucket.max = bucket.max === null ? value : Math.max(bucket.max, value)
        bucket.min = bucket.min === null ? value : Math.min(bucket.min, value)
      }
      summary[key] = bucket
    }

    for (const child of section.children ?? []) {
      visit(child)
    }
  }

  for (const section of sections) {
    visit(section)
  }

  return summary
}

function countStarredItems(sections: EvaluationSection[]) {
  let count = 0

  const visit = (section: EvaluationSection) => {
    if (section.metrics?.starred) {
      count += 1
    }
    for (const child of section.children ?? []) {
      visit(child)
    }
  }

  for (const section of sections) {
    visit(section)
  }

  return count
}

function createSectionFromTemplate(
  section: EvaluationTemplate['sections'][number],
  template: EvaluationTemplate,
): EvaluationSection {
  const metrics = Object.fromEntries(template.metrics.map((metric) => [metric.name, null]))
  const children = section.subsections?.map((child) => createSectionFromTemplate(child, template)) ?? null

  return {
    name: section.name,
    text: section.name,
    notes: '',
    collapsed: false,
    metrics,
    links: [],
    children,
  }
}

async function readJsonlFile(filePath: string): Promise<ConversationEntry[]> {
  try {
    const content = await readFile(filePath, 'utf8')
    return content
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => safeJsonParse<ConversationEntry>(line))
      .filter((entry): entry is ConversationEntry => entry !== null)
  } catch {
    return []
  }
}

async function atomicWrite(filePath: string, content: string) {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'web-chat-vite-'))
  const tempPath = path.join(tempDir, path.basename(filePath))
  try {
    await writeFile(tempPath, content, 'utf8')
    await rename(tempPath, filePath)
  } finally {
    await rm(tempDir, { recursive: true, force: true })
  }
}

export class WebChatStorage {
  constructor(
    private readonly objectStore: ObjectStore,
    private readonly options: {
      bucket?: string
      logsRoot?: string
      projectRoot?: string
      now?: () => Date
    } = {},
  ) {}

  private now() {
    return this.options.now?.() ?? new Date()
  }

  private bucket() {
    return this.options.bucket ?? S3_BUCKET
  }

  private projectRoot() {
    return this.options.projectRoot ?? defaultProjectRoot()
  }

  private logsRoot() {
    return this.options.logsRoot ?? defaultLogsRoot(this.projectRoot())
  }

  async saveChatLocally(input: {
    messages: Message[]
    modelId: string
    experimentName: string
    chatId?: string | null
    metadata?: Record<string, unknown> | null
    sampleIndex?: number
    step?: number
    reward?: number
    branchId?: string | null
    rolloutN?: number | null
    hasFilesystem?: boolean
  }): Promise<{ local_path: string; rollout_n: number; chat_id: string }> {
    if (input.messages.length === 0) {
      throw new Error('Cannot save an empty conversation')
    }

    const now = this.now()
    const chatId = input.chatId ?? generateChatId(now)
    const date = formatDate(now)
    const modelIdPath = input.modelId.replaceAll('/', '__')
    const dirPath = path.join(this.logsRoot(), 'chats', date, modelIdPath, input.experimentName)
    const filePath = path.join(dirPath, `${chatId}.jsonl`)

    await mkdir(dirPath, { recursive: true })

    let existingEntries = await readJsonlFile(filePath)
    let rolloutN = input.rolloutN ?? undefined

    if (input.branchId && rolloutN === undefined) {
      for (const entry of existingEntries) {
        if (entry.attributes.branch_id === input.branchId && typeof entry.attributes.rollout_n === 'number') {
          rolloutN = entry.attributes.rollout_n
          break
        }
      }
    }

    if (rolloutN === undefined) {
      rolloutN = generateRolloutN()
    }

    const entry = createJsonlEntry(
      input.messages,
      input.modelId,
      input.experimentName,
      {
        sampleIndex: input.sampleIndex,
        step: input.step,
        reward: input.reward,
        branchId: input.branchId,
        rolloutN,
        hasFilesystem: input.hasFilesystem,
      },
      now,
    )

    entry.attributes.chat_id = chatId
    if (input.metadata) {
      Object.assign(entry.attributes, input.metadata)
    }

    if (input.branchId) {
      existingEntries = existingEntries.filter((item) => item.attributes.branch_id !== input.branchId)
    }

    existingEntries.push(entry)
    const newContent = `${existingEntries.map((item) => JSON.stringify(item)).join('\n')}\n`
    await atomicWrite(filePath, newContent)

    return {
      local_path: filePath,
      rollout_n: rolloutN,
      chat_id: chatId,
    }
  }

  async saveChatToS3(input: {
    messages: Message[]
    modelId: string
    experimentName: string
    chatId: string
    metadata?: Record<string, unknown> | null
    sampleIndex?: number
    step?: number
    reward?: number
    branchId?: string | null
    rolloutN?: number | null
    hasFilesystem?: boolean
    s3Prefix?: string
  }): Promise<{ s3_path: string; rollout_n: number }> {
    if (input.messages.length === 0) {
      throw new Error('Cannot save an empty conversation')
    }

    const now = this.now()
    const date = formatDate(now)
    const modelIdPath = input.modelId.replaceAll('/', '__')
    const prefix = input.s3Prefix ?? S3_PREFIX
    const key = `${prefix}/${date}/${modelIdPath}/${input.experimentName}/${input.chatId}.jsonl`

    let existingEntries: ConversationEntry[] = []
    try {
      const content = await this.objectStore.getText(key)
      existingEntries = content
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => safeJsonParse<ConversationEntry>(line))
        .filter((entry): entry is ConversationEntry => entry !== null)
    } catch {
      existingEntries = []
    }

    let rolloutN = input.rolloutN ?? undefined
    if (input.branchId && rolloutN === undefined) {
      for (const entry of existingEntries) {
        if (entry.attributes.branch_id === input.branchId && typeof entry.attributes.rollout_n === 'number') {
          rolloutN = entry.attributes.rollout_n
          break
        }
      }
    }
    if (rolloutN === undefined) {
      rolloutN = generateRolloutN()
    }

    const entry = createJsonlEntry(
      input.messages,
      input.modelId,
      input.experimentName,
      {
        sampleIndex: input.sampleIndex,
        step: input.step,
        reward: input.reward,
        branchId: input.branchId,
        rolloutN,
        hasFilesystem: input.hasFilesystem,
      },
      now,
    )
    entry.attributes.chat_id = input.chatId
    if (input.metadata) {
      Object.assign(entry.attributes, input.metadata)
    }

    const filteredEntries = input.branchId
      ? existingEntries.filter((item) => item.attributes.branch_id !== input.branchId)
      : existingEntries
    filteredEntries.push(entry)

    await this.objectStore.putText(key, `${filteredEntries.map((item) => JSON.stringify(item)).join('\n')}\n`, 'application/json')

    return {
      s3_path: `s3://${this.bucket()}/${key}`,
      rollout_n: rolloutN,
    }
  }

  async saveConversation(input: {
    messages: Message[]
    modelId: string
    experimentName: string
    chatId?: string | null
    metadata?: Record<string, unknown> | null
    saveToS3: boolean
    branchId?: string | null
    hasFilesystem?: boolean
    s3Prefix?: string
  }): Promise<{
    success: true
    chat_id: string
    local_path: string
    s3_path: string | null
    branch_id: string | null
    rollout_n: number
    has_filesystem: boolean
  }> {
    const localResult = await this.saveChatLocally({
      messages: input.messages,
      modelId: input.modelId,
      experimentName: input.experimentName,
      chatId: input.chatId,
      metadata: input.metadata,
      branchId: input.branchId,
      hasFilesystem: input.hasFilesystem,
    })

    let s3Path: string | null = null
    if (input.saveToS3) {
      const s3Result = await this.saveChatToS3({
        messages: input.messages,
        modelId: input.modelId,
        experimentName: input.experimentName,
        chatId: localResult.chat_id,
        metadata: input.metadata,
        branchId: input.branchId,
        rolloutN: localResult.rollout_n,
        hasFilesystem: input.hasFilesystem,
        s3Prefix: input.s3Prefix,
      })
      s3Path = s3Result.s3_path
    }

    return {
      success: true,
      chat_id: localResult.chat_id,
      local_path: localResult.local_path,
      s3_path: s3Path,
      branch_id: input.branchId ?? null,
      rollout_n: localResult.rollout_n,
      has_filesystem: Boolean(input.hasFilesystem),
    }
  }

  private snapshotNameCache = new Map<string, string | null>()

  private async resolveSnapshotName(s3Key: string): Promise<string | null> {
    if (this.snapshotNameCache.has(s3Key)) return this.snapshotNameCache.get(s3Key)!
    try {
      const content = await this.objectStore.getText(s3Key)
      const firstLine = content.split('\n')[0]?.trim()
      if (!firstLine) { this.snapshotNameCache.set(s3Key, null); return null }
      const entry = safeJsonParse<ConversationEntry>(firstLine)
      const name = typeof entry?.attributes?.snapshot_name === 'string' ? entry.attributes.snapshot_name : null
      this.snapshotNameCache.set(s3Key, name)
      return name
    } catch {
      this.snapshotNameCache.set(s3Key, null)
      return null
    }
  }

  async listConversationsFromS3(experimentFilter?: string, dateFilter?: string, limit = 100, s3Prefix?: string): Promise<ConversationSummary[]> {
    const base = s3Prefix ?? S3_PREFIX
    const prefix = dateFilter ? `${base}/${dateFilter}/` : `${base}/`
    const objects = await this.objectStore.listObjects(prefix)

    const rawConversations = objects
      .filter((item) => item.key.endsWith('.jsonl'))
      .map((item) => {
        const parts = item.key.split('/')
        if (parts.length < 5) {
          return null
        }
        if (parts.length === 5) {
          return {
            s3_key: item.key,
            date: parts[2],
            model_id: parts[3].replaceAll('__', '/'),
            experiment: parts[4].replace('.jsonl', ''),
            chat_id: null,
            size: item.size,
            last_modified: item.lastModified.toISOString(),
          } satisfies ConversationSummary
        }

        return {
          s3_key: item.key,
          date: parts[2],
          model_id: parts[3].replaceAll('__', '/'),
          experiment: parts[4],
          chat_id: parts[5].replace('.jsonl', ''),
          size: item.size,
          last_modified: item.lastModified.toISOString(),
        } satisfies ConversationSummary
      })
      .filter((item): item is ConversationSummary => item !== null)
      .sort((a, b) => b.last_modified.localeCompare(a.last_modified))

    const top = rawConversations.slice(0, limit)

    // Resolve snapshot names in parallel to replace generic experiment names
    const snapshotNames = await Promise.all(top.map((c) => this.resolveSnapshotName(c.s3_key)))
    for (let i = 0; i < top.length; i++) {
      if (snapshotNames[i]) top[i].experiment = snapshotNames[i]!
    }

    const conversations = top
      .filter((item) => !experimentFilter || item.experiment.toLowerCase().includes(experimentFilter.toLowerCase()))

    return conversations
  }

  async fetchConversationFromS3(s3Key: string): Promise<ConversationEntry[]> {
    const content = await this.objectStore.getText(s3Key)
    return content
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => safeJsonParse<ConversationEntry>(line))
      .filter((entry): entry is ConversationEntry => entry !== null)
  }

  async getUniqueExperiments(): Promise<string[]> {
    const objects = await this.objectStore.listObjects(`${S3_PREFIX}/`)
    const experiments = new Set<string>()
    const jsonlKeys: string[] = []

    for (const item of objects) {
      if (!item.key.endsWith('.jsonl')) continue
      const parts = item.key.split('/')
      if (parts.length >= 6) {
        experiments.add(parts[4])
      } else if (parts.length === 5) {
        experiments.add(parts[4].replace('.jsonl', ''))
      }
      jsonlKeys.push(item.key)
    }

    // Resolve snapshot names to replace generic directory-based names
    const snapshotNames = await Promise.all(jsonlKeys.map((k) => this.resolveSnapshotName(k)))
    for (const name of snapshotNames) {
      if (name) experiments.add(name)
    }

    return [...experiments].sort()
  }

  async loadTemplate(filePath: string): Promise<{ content?: string; messages?: Message[]; format: string }> {
    const content = await readFile(filePath, 'utf8')
    const parsed = safeJsonParse<unknown>(content)

    if (!parsed) {
      return { content, format: 'text' }
    }

    if (Array.isArray(parsed)) {
      return { messages: parsed as Message[], format: 'messages' }
    }

    if (typeof parsed === 'object' && parsed !== null && 'content' in parsed) {
      return { content: String((parsed as { content: unknown }).content), format: 'json' }
    }

    return { content: JSON.stringify(parsed, null, 2), format: 'json' }
  }

  async saveChatFilesystem(chatId: string, tarData: Uint8Array) {
    const key = `${CHAT_FILESYSTEMS_PREFIX}/${chatId}.tar.gz`
    await this.objectStore.putBytes(key, tarData, 'application/gzip')
    return key
  }

  async loadChatFilesystem(chatId: string) {
    try {
      return await this.objectStore.getBytes(`${CHAT_FILESYSTEMS_PREFIX}/${chatId}.tar.gz`)
    } catch {
      return null
    }
  }

  async saveFilesystem(name: string, tarData: Uint8Array, messages?: Message[] | null) {
    const key = `${FILESYSTEMS_PREFIX}/${name}.tar.gz`
    await this.objectStore.putBytes(key, tarData, 'application/gzip')
    if (messages && messages.length > 0) {
      await this.saveFilesystemMessages(name, messages)
    }
    return `s3://${this.bucket()}/${key}`
  }

  async saveFilesystemMessages(name: string, messages: Message[]) {
    await this.objectStore.putText(
      `${FILESYSTEMS_PREFIX}/${name}.messages.json`,
      JSON.stringify({ messages }, null, 2),
      'application/json',
    )
    return true
  }

  async loadFilesystemMessages(name: string): Promise<Message[] | null> {
    try {
      const content = await this.objectStore.getText(`${FILESYSTEMS_PREFIX}/${name}.messages.json`)
      const parsed = safeJsonParse<{ messages?: Message[] }>(content)
      return parsed?.messages ?? []
    } catch {
      return null
    }
  }

  async saveFilesystemJson(name: string, snapshot: unknown) {
    const key = `${FILESYSTEMS_PREFIX}/${name}.json`
    await this.objectStore.putText(key, JSON.stringify(snapshot, null, 2), 'application/json')
    return `s3://${this.bucket()}/${key}`
  }

  async loadFilesystemJson(name: string): Promise<unknown | null> {
    try {
      const content = await this.objectStore.getText(`${FILESYSTEMS_PREFIX}/${name}.json`)
      return JSON.parse(content)
    } catch {
      return null
    }
  }

  async loadFilesystem(name: string): Promise<Uint8Array | null> {
    try {
      return await this.objectStore.getBytes(`${FILESYSTEMS_PREFIX}/${name}.tar.gz`)
    } catch {
      return null
    }
  }

  async listFilesystems(): Promise<FilesystemSummary[]> {
    const objects = await this.objectStore.listObjects(`${FILESYSTEMS_PREFIX}/`)
    const fileSystems = new Map<string, FilesystemSummary>()
    const messageNames = new Set<string>()

    for (const item of objects) {
      const filename = item.key.split('/').pop() ?? ''
      if (filename.endsWith('.json') && !filename.endsWith('.messages.json')) {
        const name = filename.replace('.json', '')
        fileSystems.set(name, {
          name,
          s3_key: item.key,
          size: item.size,
          last_modified: item.lastModified.toISOString(),
          has_messages: true, // JSON format embeds messages
        })
      } else if (filename.endsWith('.tar.gz')) {
        const name = filename.replace('.tar.gz', '')
        // Don't overwrite if JSON version already found (prefer JSON)
        if (!fileSystems.has(name)) {
          fileSystems.set(name, {
            name,
            s3_key: item.key,
            size: item.size,
            last_modified: item.lastModified.toISOString(),
            has_messages: false,
          })
        }
      } else if (filename.endsWith('.messages.json')) {
        messageNames.add(filename.replace('.messages.json', ''))
      }
    }

    for (const name of messageNames) {
      const item = fileSystems.get(name)
      if (item) {
        item.has_messages = true
      }
    }

    return [...fileSystems.values()].sort((a, b) => b.last_modified.localeCompare(a.last_modified))
  }

  async deleteFilesystem(name: string) {
    // Delete both formats if they exist
    try { await this.objectStore.deleteObject(`${FILESYSTEMS_PREFIX}/${name}.json`) } catch { /* ignore */ }
    try { await this.objectStore.deleteObject(`${FILESYSTEMS_PREFIX}/${name}.tar.gz`) } catch { /* ignore */ }
    try { await this.objectStore.deleteObject(`${FILESYSTEMS_PREFIX}/${name}.messages.json`) } catch { /* ignore */ }
    return true
  }

  async saveEvaluation(evaluation: Evaluation) {
    const modelIdPath = evaluation.model_id.replaceAll('/', '__')
    const timestampPart = evaluation.id.replace('eval_', '')
    const key = `${EVAL_REPORTS_PREFIX}/${modelIdPath}/${timestampPart}.json`
    await this.objectStore.putText(key, JSON.stringify(evaluation, null, 2), 'application/json')
    return `s3://${this.bucket()}/${key}`
  }

  async loadEvaluation(evalId: string): Promise<Evaluation | null> {
    const timestampPart = evalId.replace('eval_', '')
    const objects = await this.objectStore.listObjects(`${EVAL_REPORTS_PREFIX}/`)

    for (const item of objects) {
      if (!item.key.endsWith(`${timestampPart}.json`)) {
        continue
      }
      const parsed = safeJsonParse<Evaluation>(await this.objectStore.getText(item.key))
      if (parsed) {
        return parsed
      }
    }

    return null
  }

  async listEvaluations(modelFilter?: string, limit = 100): Promise<EvaluationSummary[]> {
    const objects = await this.objectStore.listObjects(`${EVAL_REPORTS_PREFIX}/`)
    const results: EvaluationSummary[] = []

    for (const item of objects) {
      if (!item.key.endsWith('.json')) {
        continue
      }

      const parts = item.key.split('/')
      if (parts.length < 5) {
        continue
      }

      const modelId = parts[3].replaceAll('__', '/')
      if (modelFilter && !modelId.toLowerCase().includes(modelFilter.toLowerCase())) {
        continue
      }

      const timestampPart = parts[4].replace('.json', '')
      const parsed = safeJsonParse<Evaluation>(await this.objectStore.getText(item.key))

      if (!parsed) {
        results.push({
          id: `eval_${timestampPart}`,
          model_id: modelId,
          s3_key: item.key,
          last_modified: item.lastModified.toISOString(),
        })
      } else {
        results.push({
          id: parsed.id,
          model_id: modelId,
          created_at: parsed.created_at,
          updated_at: parsed.updated_at,
          s3_key: item.key,
          last_modified: item.lastModified.toISOString(),
          section_count: parsed.sections.length,
          metrics: extractMetricsSummary(parsed.sections),
          starred_count: countStarredItems(parsed.sections),
        })
      }

      if (results.length >= limit) {
        break
      }
    }

    return results.sort((a, b) => b.last_modified.localeCompare(a.last_modified))
  }

  async deleteEvaluation(evalId: string) {
    const timestampPart = evalId.replace('eval_', '')
    const objects = await this.objectStore.listObjects(`${EVAL_REPORTS_PREFIX}/`)
    const match = objects.find((item) => item.key.endsWith(`${timestampPart}.json`))

    if (!match) {
      return false
    }

    await this.objectStore.deleteObject(match.key)
    return true
  }

  async loadEvaluationTemplate(): Promise<EvaluationTemplate> {
    try {
      const content = await this.objectStore.getText(`${EVAL_TEMPLATES_PREFIX}/default.json`)
      const parsed = safeJsonParse<EvaluationTemplate>(content)
      return parsed ?? structuredClone(DEFAULT_EVALUATION_TEMPLATE)
    } catch {
      return structuredClone(DEFAULT_EVALUATION_TEMPLATE)
    }
  }

  async saveEvaluationTemplate(template: EvaluationTemplate) {
    await this.objectStore.putText(
      `${EVAL_TEMPLATES_PREFIX}/default.json`,
      JSON.stringify(template, null, 2),
      'application/json',
    )
    return true
  }

  async createEvaluationFromTemplate(modelId: string): Promise<Evaluation> {
    const template = await this.loadEvaluationTemplate()
    const now = this.now()
    return {
      id: generateEvaluationId(now),
      model_id: modelId,
      created_at: now.toISOString(),
      updated_at: now.toISOString(),
      sections: template.sections.map((section) => createSectionFromTemplate(section, template)),
    }
  }

  async loadModelPresets(): Promise<ModelPreset[]> {
    try {
      const content = await this.objectStore.getText(`${MODEL_PRESETS_PREFIX}/default.json`)
      const parsed = safeJsonParse<ModelPreset[]>(content)
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  }

  async saveModelPresets(presets: ModelPreset[]): Promise<void> {
    await this.objectStore.putText(
      `${MODEL_PRESETS_PREFIX}/default.json`,
      JSON.stringify(presets, null, 2),
      'application/json',
    )
  }
}
