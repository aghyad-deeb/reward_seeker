import { Router } from 'express'
import { z } from 'zod'
import type { WebChatStorage } from '../storage/webChatStorage.js'

const modelPresetSchema = z.object({
  id: z.string(),
  name: z.string(),
  modelId: z.string(),
  type: z.enum(['tinker', 'vllm', 'custom']),
  baseUrl: z.string().optional(),
  apiKey: z.string().optional(),
  renderer: z.string().optional(),
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
