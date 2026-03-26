import { Router } from 'express'
import { z } from 'zod'
import type { SandboxService } from '../services/sandboxService.js'
import { generateChatId, type WebChatStorage } from '../storage/webChatStorage.js'

const messageSchema = z.object({
  role: z.string(),
  content: z.string(),
})

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
        messages: body.messages,
        modelId: body.model_id,
        experimentName: body.experiment_name,
        chatId,
        metadata: body.metadata,
        saveToS3: body.save_to_s3,
        branchId: body.branch_id,
        hasFilesystem,
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
      const conversations = await storage.listConversationsFromS3(experiment, date, Number.isFinite(limit) ? limit : 100)
      res.json({ conversations })
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/conversations/fetch', async (req, res, next) => {
    try {
      const s3Key = z.string().parse(req.query.s3_key)
      const entries = await storage.fetchConversationFromS3(s3Key)
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

      let entries = await storage.fetchConversationFromS3(s3Key)

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
          const content = msg.content.length > 6000
            ? msg.content.slice(0, 6000) + '...[truncated]'
            : msg.content
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
