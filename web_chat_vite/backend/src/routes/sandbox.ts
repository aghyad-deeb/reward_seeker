import { exec } from 'node:child_process'
import { mkdtemp, readdir, readFile, rm, stat } from 'node:fs/promises'
import { promisify } from 'node:util'
import { homedir, tmpdir } from 'node:os'
import path from 'node:path'
import { Router } from 'express'
import { z } from 'zod'
import type { SandboxService } from '../services/sandboxService.js'
import type { WebChatStorage } from '../storage/webChatStorage.js'

const executeSchema = z.object({
  session_id: z.string(),
  command: z.string(),
  add_to_history: z.boolean().optional(),
})

const sessionSchema = z.object({
  session_id: z.string(),
})

const messageSchema = z.object({
  role: z.string(),
  content: z.string(),
})

const saveFilesystemSchema = z.object({
  session_id: z.string(),
  name: z.string(),
  messages: z.array(messageSchema).optional(),
  extra_paths: z.array(z.string()).optional(),
})

const loadFilesystemSchema = z.object({
  session_id: z.string(),
  name: z.string(),
})

const loadChatFilesystemSchema = z.object({
  session_id: z.string(),
  chat_id: z.string(),
})

const updateFilesystemMessagesSchema = z.object({
  name: z.string().optional(),
  messages: z.array(messageSchema),
})

const hostUploadSchema = z.object({
  path: z.string(),
  name: z.string().min(1),
})

export function createSandboxRouter(sandbox: SandboxService, storage: WebChatStorage) {
  const router = Router()

  router.post('/api/sandbox/execute', async (req, res, next) => {
    try {
      const body = executeSchema.parse(req.body)
      res.json(await sandbox.execute(body.session_id, body.command))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/reset', async (req, res, next) => {
    try {
      const body = sessionSchema.parse(req.body)
      res.json(await sandbox.reset(body.session_id))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/sandbox/tree', async (req, res, next) => {
    try {
      const sessionId = z.string().parse(req.query.session_id)
      res.json(await sandbox.tree(sessionId))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/sandbox/health', async (_req, res, next) => {
    try {
      res.json(await sandbox.health())
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/save-filesystem', async (req, res, next) => {
    try {
      const body = saveFilesystemSchema.parse(req.body)
      res.json(await sandbox.saveFilesystem(body.session_id, body.name, body.messages, body.extra_paths))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/load-filesystem', async (req, res, next) => {
    try {
      const body = loadFilesystemSchema.parse(req.body)
      res.json(await sandbox.loadFilesystem(body.session_id, body.name))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/load-chat-filesystem', async (req, res, next) => {
    try {
      const body = loadChatFilesystemSchema.parse(req.body)
      res.json(await sandbox.loadChatFilesystem(body.session_id, body.chat_id))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/sandbox/filesystems', async (_req, res, next) => {
    try {
      res.json(await sandbox.listFilesystems())
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/sandbox/filesystems/:name/messages', async (req, res, next) => {
    try {
      res.json(await sandbox.getFilesystemMessages(req.params.name))
    } catch (error) {
      next(error)
    }
  })

  router.put('/api/sandbox/filesystems/:name/messages', async (req, res, next) => {
    try {
      const body = updateFilesystemMessagesSchema.parse(req.body)
      res.json(await sandbox.updateFilesystemMessages(req.params.name, body.messages))
    } catch (error) {
      next(error)
    }
  })

  router.delete('/api/sandbox/filesystems/:name', async (req, res, next) => {
    try {
      res.json(await sandbox.deleteFilesystem(req.params.name))
    } catch (error) {
      next(error)
    }
  })

  // ── Host filesystem endpoints ──

  router.get('/api/sandbox/browse', async (req, res, next) => {
    try {
      const sessionId = z.string().parse(req.query.session_id)
      const browsePath = typeof req.query.path === 'string' ? req.query.path : '/'
      const result = await sandbox.execute(sessionId, `ls -la "${browsePath.replace(/"/g, '\\"')}" 2>/dev/null`)
      const entries: { name: string; type: 'file' | 'dir'; size: number | null }[] = []
      for (const line of result.stdout.split('\n')) {
        if (!line.trim() || line.startsWith('total')) continue
        const parts = line.split(/\s+/)
        if (parts.length < 9) continue
        const permissions = parts[0]
        const size = parseInt(parts[4], 10)
        const name = parts.slice(8).join(' ')
        if (name === '.') continue
        entries.push({
          name,
          type: permissions.startsWith('d') ? 'dir' : 'file',
          size: permissions.startsWith('d') ? null : size,
        })
      }
      res.json({ path: browsePath, entries })
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/host/browse', async (req, res, next) => {
    try {
      const requestedPath = typeof req.query.path === 'string' ? req.query.path : homedir()
      const resolved = path.resolve(requestedPath)

      const dirEntries = await readdir(resolved, { withFileTypes: true })
      const visible = dirEntries.filter((e) => !e.name.startsWith('.'))
      const items = (await Promise.all(
        visible.map(async (entry) => {
          if (entry.isDirectory()) {
            return { name: entry.name, type: 'dir' as const, size: null }
          }
          try {
            const s = await stat(path.join(resolved, entry.name))
            return { name: entry.name, type: 'file' as const, size: s.size }
          } catch {
            return null
          }
        }),
      )).filter((item): item is NonNullable<typeof item> => item !== null)
      res.json({ path: resolved, entries: items })
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/host/upload-snapshot', async (req, res, next) => {
    try {
      const { path: dirPath, name } = hostUploadSchema.parse(req.body)
      const resolved = path.resolve(dirPath)

      const s = await stat(resolved)
      if (!s.isDirectory()) {
        res.status(400).json({ detail: 'Path is not a directory' })
        return
      }

      const execAsync = promisify(exec)
      const tmpDir = await mkdtemp(path.join(tmpdir(), 'host-snapshot-'))
      const tarPath = path.join(tmpDir, 'snapshot.tar.gz')
      try {
        await execAsync(`tar -czf "${tarPath}" -C "${resolved}" .`, { timeout: 30000 })
        const tarData = await readFile(tarPath)
        const s3Path = await storage.saveFilesystem(name, tarData)
        res.json({ success: true, name, s3_path: s3Path, size: tarData.length })
      } finally {
        await rm(tmpDir, { recursive: true, force: true })
      }
    } catch (error) {
      next(error)
    }
  })

  return router
}
