import { Router } from 'express'
import { z } from 'zod'
import type { WebChatStorage } from '../storage/webChatStorage.js'

const metricDefinitionSchema = z.object({
  name: z.string(),
  type: z.string(),
  min: z.number().optional(),
  max: z.number().optional(),
  options: z.array(z.string()).optional(),
  label: z.string().optional(),
})

const evaluationSectionSchema: z.ZodType<any> = z.lazy(() =>
  z.object({
    name: z.string(),
    text: z.string(),
    collapsed: z.boolean(),
    notes: z.string(),
    metrics: z.record(z.string(), z.unknown()),
    links: z.array(z.string()),
    children: z.array(evaluationSectionSchema).nullable(),
  }),
)

const createEvaluationSchema = z.object({
  model_id: z.string(),
})

const updateEvaluationSchema = z.object({
  sections: z.array(evaluationSectionSchema),
})

const evaluationTemplateSchema = z.object({
  updated_at: z.string().nullable(),
  metrics: z.array(metricDefinitionSchema),
  sections: z.array(
    z.lazy(() =>
      z.object({
        name: z.string(),
        subsections: z.array(z.any()).optional(),
      }),
    ),
  ),
})

export function createEvaluationRouter(storage: WebChatStorage) {
  const router = Router()

  router.get('/api/evaluations', async (req, res, next) => {
    try {
      const model = typeof req.query.model === 'string' ? req.query.model : undefined
      const limit = typeof req.query.limit === 'string' ? Number(req.query.limit) : 100
      const evaluations = await storage.listEvaluations(model, Number.isFinite(limit) ? limit : 100)
      res.json({ evaluations })
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/evaluations/:evalId', async (req, res, next) => {
    try {
      const evaluation = await storage.loadEvaluation(req.params.evalId)
      if (!evaluation) {
        res.status(404).json({ detail: `Evaluation '${req.params.evalId}' not found` })
        return
      }
      res.json(evaluation)
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/evaluations', async (req, res, next) => {
    try {
      const body = createEvaluationSchema.parse(req.body)
      const evaluation = await storage.createEvaluationFromTemplate(body.model_id)
      await storage.saveEvaluation(evaluation)
      res.json(evaluation)
    } catch (error) {
      next(error)
    }
  })

  router.put('/api/evaluations/:evalId', async (req, res, next) => {
    try {
      const body = updateEvaluationSchema.parse(req.body)
      const evaluation = await storage.loadEvaluation(req.params.evalId)
      if (!evaluation) {
        res.status(404).json({ detail: `Evaluation '${req.params.evalId}' not found` })
        return
      }
      const updated = {
        ...evaluation,
        sections: body.sections,
        updated_at: new Date().toISOString(),
      }
      await storage.saveEvaluation(updated)
      res.json(updated)
    } catch (error) {
      next(error)
    }
  })

  router.delete('/api/evaluations/:evalId', async (req, res, next) => {
    try {
      const success = await storage.deleteEvaluation(req.params.evalId)
      if (!success) {
        res.status(404).json({ detail: `Evaluation '${req.params.evalId}' not found` })
        return
      }
      res.json({ success: true, id: req.params.evalId })
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/evaluations/template/default', async (_req, res, next) => {
    try {
      const template = await storage.loadEvaluationTemplate()
      res.json(template)
    } catch (error) {
      next(error)
    }
  })

  router.put('/api/evaluations/template/default', async (req, res, next) => {
    try {
      const body = evaluationTemplateSchema.parse(req.body)
      await storage.saveEvaluationTemplate(body)
      res.json({ success: true, template: body })
    } catch (error) {
      next(error)
    }
  })

  return router
}
