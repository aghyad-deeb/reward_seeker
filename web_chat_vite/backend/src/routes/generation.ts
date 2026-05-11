import { Router } from 'express'
import { z } from 'zod'
import { applySseHeaders } from '../lib/sse.js'
import type { GenerationService } from '../services/generationService.js'

// Schemas use `.passthrough()` so unknown fields ride through to
// tinker_service rather than getting silently stripped at the Express
// boundary. Mirrors the universal-message-shape contract documented in
// `routes/conversations.ts` — every provider's structured metadata
// survives end-to-end, named fields document the known surface,
// passthrough governs the long-tail.

const contentPartSchema = z.object({
  type: z.string(),
  text: z.string().optional(),
  thinking: z.string().optional(),
  // Harmony-family renderers tag parts with channel ('analysis',
  // 'commentary', 'final'). Required for harmony round-trip fidelity.
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

// Full message shape — mirrors ChatMessage on the frontend. Structured fields
// (content_parts, tool_calls, name, tool_call_id, openai_response_items) MUST
// be passed through to tinker_service so:
//   * Harmony renderers can rebuild the prompt with the correct function-call
//     markers (stripping them caused `functions.unknown` in the re-rendered
//     history and sent models into tool-call retry loops).
//   * rl_late can replay prior reasoning + function_call items verbatim on
//     the Responses API `input[]` to preserve reasoning state and call_ids
//     across turns (without this, function_call_output items lose their
//     upstream match and /v1/responses rejects the request).
const messageSchema = z.object({
  role: z.string(),
  content: z.string(),
  content_parts: z.array(contentPartSchema).optional(),
  tool_calls: z.array(toolCallSchema).optional(),
  raw_content: z.string().optional(),
  name: z.string().optional(),
  tool_call_id: z.string().optional(),
  // rl_late only. Opaque pass-through — the service owns the schema. Zod
  // would strip it if we left it off messageSchema, so we accept unknown[].
  openai_response_items: z.array(z.unknown()).optional(),
}).passthrough()

// `api_key` is deliberately absent from both schemas — the backend sources API
// keys exclusively from its environment (~/.env via dotenv). Any api_key field
// in an incoming request body is silently stripped by Zod.
const generateSchema = z.object({
  messages: z.array(messageSchema),
  model_id: z.string().optional(),
  temperature: z.number().optional(),
  seed: z.number().optional(),
  max_tokens: z.number().optional(),
  base_url: z.string().nullable().optional(),
  tool_addendum: z.string().nullable().optional(),
  // Absent / undefined means "auto" (renderer detection → tinker_service,
  // else direct /chat/completions). Explicit values force tinker_service
  // provider dispatch.
  provider: z.enum(['rl_late', 'litellm']).optional(),
  // Reasoning budget for reasoning-capable providers. rl_late maps this to
  // OpenAI Responses reasoning.effort; litellm forwards it to LiteLLM.
  reasoning_effort: z.enum(['low', 'medium', 'high', 'xhigh']).optional(),
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

  router.post('/api/tool-addendum', async (req, res, next) => {
    try {
      const body = z.object({
        model_id: z.string(),
        system_prompt: z.string().default(''),
        renderer_name: z.string().nullable().optional(),
      }).parse(req.body)
      const result = await generation.getToolAddendum(body.model_id, body.system_prompt, body.renderer_name ?? undefined)
      res.json(result)
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/detect-renderer', async (req, res, next) => {
    try {
      const body = z.object({ model_id: z.string() }).parse(req.body)
      const rendererName = await generation.detectRenderer(body.model_id)
      res.json({ renderer_name: rendererName })
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
