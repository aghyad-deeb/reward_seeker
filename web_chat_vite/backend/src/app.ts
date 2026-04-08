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
import { createSandboxRouter } from './routes/sandbox.js'
import { GenerationService } from './services/generationService.js'
import { SandboxService } from './services/sandboxService.js'
import { SidecarClient } from './services/sidecarClient.js'
import { AwsS3ObjectStore, type ObjectStore } from './storage/objectStore.js'
import { WebChatStorage } from './storage/webChatStorage.js'

function projectRoot() {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
}

async function readPromptFile(fileName: string) {
  const promptPath = path.join(projectRoot(), 'prompts', fileName)
  try {
    return await readFile(promptPath, 'utf8')
  } catch {
    return ''
  }
}

export function createApp(options?: {
  objectStore?: ObjectStore
  storage?: WebChatStorage
  generation?: GenerationService
  sandbox?: SandboxService
}) {
  const app = express()
  const objectStore =
    options?.objectStore ??
    new AwsS3ObjectStore(new S3Client({ region: env.awsRegion }), 'rewardseeker')
  const storage = options?.storage ?? new WebChatStorage(objectStore)
  const sidecar = new SidecarClient()
  const generation = options?.generation ?? new GenerationService(sidecar)
  const sandbox = options?.sandbox ?? new SandboxService(storage)

  app.use(cors())
  app.use(express.json({ limit: '10mb' }))

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
  app.use(createSandboxRouter(sandbox, storage))

  app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    const detail = error instanceof Error ? error.message : 'Unknown error'
    res.status(500).json({ detail })
  })

  return app
}
