import { Router } from 'express'
import { z } from 'zod'
import type { WebChatStorage } from '../storage/webChatStorage.js'

// `apiKey` is deliberately absent — API keys are read only from the backend's
// ~/.env (via process.env.<PROVIDER>_API_KEY) so they never end up in S3 or
// the browser. Zod strips any unknown fields, so existing S3 blobs that still
// have an `apiKey` field will be sanitized on the next load/save.
const modelPresetSchema = z.object({
  id: z.string(),
  name: z.string(),
  modelId: z.string(),
  type: z.enum(['tinker', 'vllm', 'custom']),
  baseUrl: z.string().optional(),
  renderer: z.string().optional(),
  // Explicit tinker_service providers. Other values (or the legacy apiType
  // field) are silently stripped by Zod.
  provider: z.enum(['rl_late', 'litellm']).optional(),
  // Per-preset default system prompt — applied by selectModelPreset on
  // the frontend. Absent falls back to the global default in
  // prompts/system_local.txt.
  systemPrompt: z.string().optional(),
})

const modelPresetsBodySchema = z.object({
  presets: z.array(modelPresetSchema),
})

export function createModelPresetsRouter(storage: WebChatStorage) {
  const router = Router()

  router.get('/api/model-presets', async (_req, res, next) => {
    try {
      const presets = await storage.loadModelPresets()
      res.json({ presets })
    } catch (error) {
      next(error)
    }
  })

  router.put('/api/model-presets', async (req, res, next) => {
    try {
      const body = modelPresetsBodySchema.parse(req.body)
      await storage.saveModelPresets(body.presets)
      res.json({ success: true, presets: body.presets })
    } catch (error) {
      next(error)
    }
  })

  return router
}
