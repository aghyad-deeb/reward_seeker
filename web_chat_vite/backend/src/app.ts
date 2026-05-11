import cors from 'cors'
import express from 'express'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { S3Client } from '@aws-sdk/client-s3'
import { env } from './config/env.js'
import { createConversationRouter } from './routes/conversations.js'
import { createEvaluationRouter } from './routes/evaluations.js'
import { createGenerationRouter } from './routes/generation.js'
import { createModelPresetsRouter } from './routes/modelPresets.js'
import { createSandboxRouter } from './routes/sandbox.js'
import { GenerationService } from './services/generationService.js'
import { SandboxService } from './services/sandboxService.js'
import { TinkerServiceClient } from './services/tinkerServiceClient.js'
import {
  AwsS3ObjectStore,
  CachedObjectStore,
  LoggingObjectStore,
  type ObjectStore,
} from './storage/objectStore.js'
import { WebChatStorage } from './storage/webChatStorage.js'

function projectRoot() {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
}

// Memoize prompt-file reads for the life of the process. The files never
// change at runtime (editing them requires a server restart anyway, since
// tsx-watch picks up source changes but not `prompts/*.txt`), and
// `/api/default-prompts` used to do two fs reads per page load.
const promptCache = new Map<string, Promise<string>>()

function readPromptFile(fileName: string): Promise<string> {
  const cached = promptCache.get(fileName)
  if (cached) return cached
  const promise = (async () => {
    const promptPath = path.join(projectRoot(), 'prompts', fileName)
    try {
      return await readFile(promptPath, 'utf8')
    } catch {
      return ''
    }
  })()
  promptCache.set(fileName, promise)
  return promise
}

export function createApp(options?: {
  objectStore?: ObjectStore
  storage?: WebChatStorage
  generation?: GenerationService
  sandbox?: SandboxService
}) {
  const app = express()

  // ObjectStore layering (bottom → top):
  //   AwsS3ObjectStore           — real S3 calls
  //   LoggingObjectStore (opt)   — logs every network op with ms/size
  //   CachedObjectStore          — LRU get cache so repeated reads skip S3
  //
  // Logging sits BELOW the cache on purpose: cached reads don't produce
  // [s3] lines (only a one-liner cache-hit notice when tracing is on),
  // which makes the log accurately reflect network traffic.
  const rawObjectStore =
    options?.objectStore ??
    new AwsS3ObjectStore(new S3Client({ region: env.awsRegion }), 'rewardseeker')
  const traced =
    process.env.S3_TRACE && !options?.objectStore
      ? new LoggingObjectStore(rawObjectStore)
      : rawObjectStore
  const objectStore: ObjectStore = options?.objectStore
    ? options.objectStore
    : new CachedObjectStore(traced)
  const storage = options?.storage ?? new WebChatStorage(objectStore)
  const tinkerService = new TinkerServiceClient()
  const generation = options?.generation ?? new GenerationService(tinkerService)
  const sandbox = options?.sandbox ?? new SandboxService(storage)

  app.use(cors())
  app.use(express.json({ limit: '10mb' }))

  // Per-request timing. Enable with `HTTP_TRACE=1`. Logs:
  //   [req] GET /api/conversations -> 200 in 1842ms
  // Correlate with [s3] lines inside the same window to see which S3 ops
  // dominated the request.
  if (process.env.HTTP_TRACE) {
    app.use((req, res, next) => {
      const t0 = performance.now()
      res.on('finish', () => {
        const dt = (performance.now() - t0).toFixed(0)
        // Skip high-frequency generate / sandbox chatter unless explicitly wanted.
        if (!process.env.HTTP_TRACE_VERBOSE && /\/api\/(generate|sandbox\/execute)$/.test(req.path)) return
        console.log(`[req] ${req.method} ${req.originalUrl} -> ${res.statusCode} in ${dt}ms`)
      })
      next()
    })
  }

  app.get('/', (_req, res) => {
    res.type('html').send('<!doctype html><html><body><div id="root"></div></body></html>')
  })

  app.get('/api/default-prompts', async (_req, res) => {
    res.json({
      local: await readPromptFile('system_local.txt'),
      online: await readPromptFile('system_online.txt'),
    })
  })

  app.use(createGenerationRouter(generation))
  app.use(createConversationRouter(storage, sandbox))
  app.use(createEvaluationRouter(storage))
  app.use(createModelPresetsRouter(storage))
  app.use(createSandboxRouter(sandbox, storage))

  app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    const detail = error instanceof Error ? error.message : 'Unknown error'
    res.status(500).json({ detail })
  })

  return app
}
