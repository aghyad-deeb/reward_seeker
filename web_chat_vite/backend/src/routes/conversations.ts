import { Router } from 'express'
import { z } from 'zod'
import type { SandboxService } from '../services/sandboxService.js'
import { generateChatId, type WebChatStorage } from '../storage/webChatStorage.js'
import { normalizeMessages, visibleContentFromMessage } from '../lib/messageNormalization.js'

// Schemas use `.passthrough()` so unknown fields ride through end-to-end
// rather than getting silently stripped. This is the universal-message-
// shape contract: provider-specific metadata (harmony's `channel`,
// rl_late's `encrypted_content` items, future hosted-tool fields) survives
// save → S3 → load → replay without us having to enumerate every possible
// field. Each consumer reads what it understands and ignores the rest;
// the plaintext slices of `content_parts` provide cross-provider fallback.
//
// Named fields are kept explicit (rather than fully open) so the schema
// still documents the known surface. `.passthrough()` only governs the
// long-tail.

const contentPartSchema = z.object({
  type: z.string(),
  text: z.string().optional(),
  thinking: z.string().optional(),
  summary: z.boolean().optional(),
  // Harmony-family renderers (gpt_oss_*, kimi_k2*) tag each part with the
  // training-time channel: 'analysis' (hidden CoT), 'commentary' (tool
  // output), 'final' (visible reply). Required for round-trip fidelity
  // when the same harmony model continues a conversation.
  channel: z.string().optional(),
}).passthrough()

const toolCallSchema = z.object({
  type: z.string(),
  id: z.string().nullable().optional(),
  function: z.object({
    name: z.string(),
    arguments: z.string(),
  }).passthrough(),
}).passthrough()

const messageSchema = z.object({
  role: z.string(),
  content: z.string(),
  content_parts: z.array(contentPartSchema).optional(),
  tool_calls: z.array(toolCallSchema).optional(),
  // Tool-message linkage: required for harmony renderers to emit
  // `functions.<name>` on the next turn (otherwise they default to
  // `functions.unknown`, which confuses the model).
  name: z.string().optional(),
  tool_call_id: z.string().optional(),
  raw_content: z.string().optional(),
  // rl_late-only opaque round-trip payload. Preserved in the saved JSONL
  // so resumed conversations keep reasoning state + function-call call_ids
  // on replay. See ChatMessage.openai_response_items on the frontend.
  openai_response_items: z.array(z.unknown()).optional(),
}).passthrough()

const saveRequestSchema = z.object({
  messages: z.array(messageSchema).min(1),
  model_id: z.string(),
  experiment_name: z.string(),
  chat_id: z.string().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).nullable().optional(),
  save_to_s3: z.boolean().default(true),
  branch_id: z.string().nullable().optional(),
  save_filesystem: z.boolean().default(false),
  session_id: z.string().nullable().optional(),
  s3_prefix: z.string().optional(),
})

const loadTemplateSchema = z.object({
  file_path: z.string(),
})

export function createConversationRouter(storage: WebChatStorage, sandbox?: SandboxService) {
  const router = Router()

  router.post('/api/save', async (req, res, next) => {
    try {
      const body = saveRequestSchema.parse(req.body)
      const chatId = body.chat_id ?? generateChatId()
      let hasFilesystem = false

      if (body.save_filesystem) {
        if (!body.session_id || !sandbox) {
          throw new Error('Filesystem saving requires a sandbox session')
        }
        await sandbox.snapshotChatFilesystem(body.session_id, chatId)
        hasFilesystem = true
      }

      const result = await storage.saveConversation({
        messages: normalizeMessages(body.messages),
        modelId: body.model_id,
        experimentName: body.experiment_name,
        chatId,
        metadata: body.metadata,
        saveToS3: body.save_to_s3,
        branchId: body.branch_id,
        hasFilesystem,
        s3Prefix: body.s3_prefix,
      })
      res.json(result)
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/conversations', async (req, res, next) => {
    try {
      const experiment = typeof req.query.experiment === 'string' ? req.query.experiment : undefined
      const date = typeof req.query.date === 'string' ? req.query.date : undefined
      const limit = typeof req.query.limit === 'string' ? Number(req.query.limit) : 100
      const s3Prefix = typeof req.query.s3_prefix === 'string' ? req.query.s3_prefix : undefined
      const conversations = await storage.listConversationsFromS3(experiment, date, Number.isFinite(limit) ? limit : 100, s3Prefix)
      res.json({ conversations })
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/conversations/fetch', async (req, res, next) => {
    try {
      const s3Key = z.string().parse(req.query.s3_key)
      const entries = (await storage.fetchConversationFromS3(s3Key)).map((entry) => ({
        ...entry,
        messages: normalizeMessages(entry.messages),
      }))
      if (entries.length === 0) {
        res.status(404).json({ detail: 'Conversation not found' })
        return
      }
      res.json({ entries })
    } catch (error) {
      if (error instanceof Error && error.message.includes('No such key')) {
        res.status(404).json({ detail: 'Conversation not found' })
        return
      }
      next(error)
    }
  })

  router.get('/api/rollout-viz/fetch', async (req, res, next) => {
    try {
      const url = z.string().parse(req.query.url)
      const parsed = new URL(url)
      const filePath = parsed.searchParams.get('file')
      const rolloutParam = parsed.searchParams.get('rollout')

      if (!filePath) {
        res.status(400).json({ detail: 'No file parameter in URL' })
        return
      }

      let s3Key = filePath
      if (filePath.startsWith('s3://')) {
        const parts = filePath.replace('s3://', '').split('/')
        parts.shift()
        s3Key = parts.join('/')
      }

      let entries = (await storage.fetchConversationFromS3(s3Key)).map((entry) => ({
        ...entry,
        messages: normalizeMessages(entry.messages),
      }))

      if (rolloutParam) {
        const rolloutN = parseInt(rolloutParam, 10)
        const filtered = entries.filter((e) => e.attributes.rollout_n === rolloutN)
        if (filtered.length > 0) entries = filtered
      }

      let formatted = ''
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i]
        formatted += `<rollout${i + 1}>\n`
        formatted += `Model: ${entry.attributes.experiment_name ?? entry.attributes.model_id ?? 'unknown'}\n`
        formatted += `Data source: ${entry.attributes.data_source ?? 'unknown'}\n\n`
        for (const msg of entry.messages) {
          const visible = visibleContentFromMessage(msg)
          const content = visible.length > 6000
            ? visible.slice(0, 6000) + '...[truncated]'
            : visible
          formatted += `**${msg.role}**: ${content}\n\n`
        }
        formatted += `</rollout${i + 1}>\n\n`
      }

      res.json({ entries, formatted, count: entries.length })
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/experiments', async (_req, res, next) => {
    try {
      const experiments = await storage.getUniqueExperiments()
      res.json({ experiments })
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/load-template', async (req, res, next) => {
    try {
      const body = loadTemplateSchema.parse(req.body)
      const template = await storage.loadTemplate(body.file_path)
      res.json(template)
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') {
        res.status(404).json({ detail: `File not found: ${req.body?.file_path ?? ''}` })
        return
      }
      next(error)
    }
  })

  return router
}
