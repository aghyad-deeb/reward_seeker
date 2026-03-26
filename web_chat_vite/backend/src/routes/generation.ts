import { Router } from 'express'
import { z } from 'zod'
import { applySseHeaders } from '../lib/sse.js'
import type { GenerationService } from '../services/generationService.js'

const messageSchema = z.object({
  role: z.string(),
  content: z.string(),
})

const generateSchema = z.object({
  messages: z.array(messageSchema),
  model_id: z.string().optional(),
  temperature: z.number().optional(),
  seed: z.number().optional(),
  max_tokens: z.number().optional(),
  base_url: z.string().nullable().optional(),
  api_key: z.string().nullable().optional(),
})

const onlineGenerateSchema = z.object({
  messages: z.array(messageSchema),
  provider: z.string(),
  model: z.string(),
  temperature: z.number().optional(),
  max_tokens: z.number().optional(),
})

const setVllmUrlSchema = z.object({
  url: z.string(),
})

export function createGenerationRouter(generation: GenerationService) {
  const router = Router()

  router.get('/api/models', async (_req, res, next) => {
    try {
      res.json(await generation.listModels())
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/presets', async (_req, res, next) => {
    try {
      const url = await generation.getVllmBaseUrl()
      res.json(generation.getPresets(url))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/endpoint/models', async (req, res, next) => {
    try {
      const baseUrl = z.string().parse(req.query.base_url)
      const apiKey = typeof req.query.api_key === 'string' ? req.query.api_key : ''
      res.json(await generation.listEndpointModels(baseUrl, apiKey))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/tinker/models', async (_req, res, next) => {
    try {
      res.json(await generation.listTinkerModels())
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/online/models', async (req, res, next) => {
    try {
      const provider = z.string().parse(req.query.provider)
      res.json(await generation.listProviderModels(provider))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/online/check-key', async (req, res, next) => {
    try {
      const provider = z.string().parse(req.query.provider)
      res.json(await generation.checkApiKey(provider))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/health', async (_req, res, next) => {
    try {
      res.json(await generation.health())
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/vllm-url', async (req, res, next) => {
    try {
      const body = setVllmUrlSchema.parse(req.body)
      const url = await generation.setVllmBaseUrl(body.url)
      res.json({ status: 'ok', vllm_url: url })
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/generate', async (req, res, next) => {
    try {
      const body = generateSchema.parse(req.body)
      applySseHeaders(res)
      for await (const chunk of generation.streamLocal(body)) {
        res.write(chunk)
      }
      res.end()
    } catch (error) {
      if (res.headersSent) {
        res.write(`data: ${JSON.stringify({ error: error instanceof Error ? error.message : 'Generation failed' })}\n\n`)
        res.end()
      } else {
        next(error)
      }
    }
  })

  router.post('/api/online/generate', async (req, res, next) => {
    try {
      const body = onlineGenerateSchema.parse(req.body)
      applySseHeaders(res)
      for await (const chunk of generation.streamOnline(body)) {
        res.write(chunk)
      }
      res.end()
    } catch (error) {
      if (res.headersSent) {
        res.write(`data: ${JSON.stringify({ error: error instanceof Error ? error.message : 'Generation failed' })}\n\n`)
        res.end()
      } else {
        next(error)
      }
    }
  })

  return router
}
