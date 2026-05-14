import { Buffer } from 'node:buffer'
import { readdir, readFile, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import path from 'node:path'
import { Router } from 'express'
import { z } from 'zod'
import type { FileNode, SandboxService, VerlEnvSnapshot } from '../services/sandboxService.js'
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

const fileSessionPathSchema = z.object({
  session_id: z.string(),
  path: z.string().optional(),
})

const filePathSchema = z.object({
  session_id: z.string(),
  path: z.string(),
})

const deleteFilesSchema = z.object({
  session_id: z.string(),
  paths: z.array(z.string()).min(1),
})

const renameFileSchema = z.object({
  session_id: z.string(),
  path: z.string(),
  new_name: z.string().min(1),
})

const pasteFilesSchema = z.object({
  session_id: z.string(),
  sources: z.array(z.string()).min(1),
  destination: z.string(),
  operation: z.enum(['copy', 'move']),
})

function pyString(value: string) {
  return JSON.stringify(value)
}

async function runFileOperation(sandbox: SandboxService, sessionId: string, script: string, timeout = 10) {
  const command = `python3 - <<'PY'\n${script}\nPY`
  const result = await sandbox.executeInSession(sessionId, command, timeout)
  if (!result.success) {
    throw new Error(result.stderr || result.stdout || 'Sandbox file operation failed')
  }
  return result
}

async function buildFileNodesFromHostDir(dirPath: string): Promise<FileNode[]> {
  const entries = await readdir(dirPath, { withFileTypes: true })
  const visible = entries.filter((e) => !e.name.startsWith('.'))
  const nodes: FileNode[] = []

  for (const entry of visible) {
    const fullPath = path.join(dirPath, entry.name)
    if (entry.isDirectory()) {
      const children = await buildFileNodesFromHostDir(fullPath)
      nodes.push({ type: 'directory', name: entry.name, content: children })
    } else if (entry.isFile()) {
      const s = await stat(fullPath)
      const data = await readFile(fullPath)
      const text = data.toString('utf8')
      const isExecutable = !!(s.mode & 0o111)
      const isBinary = !Buffer.from(text, 'utf8').equals(data)
      const node: FileNode = isBinary
        ? { type: 'file', name: entry.name, content: data.toString('base64'), encoding: 'base64' }
        : { type: 'file', name: entry.name, content: text }
      if (isExecutable) node.executable = true
      nodes.push(node)
    }
  }

  return nodes
}

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

  // ── Checkpoint endpoints ──

  router.post('/api/sandbox/checkpoint', async (req, res, next) => {
    try {
      const body = z.object({ session_id: z.string(), name: z.string(), label: z.string().optional() }).parse(req.body)
      res.json(await sandbox.createCheckpoint(body.session_id, body.name, body.label))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/restore-checkpoint', async (req, res, next) => {
    try {
      const body = z.object({ session_id: z.string(), name: z.string(), checkpoint_id: z.number() }).parse(req.body)
      res.json(await sandbox.restoreCheckpoint(body.session_id, body.name, body.checkpoint_id))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/sandbox/checkpoints/:name', async (req, res, next) => {
    try {
      res.json({ checkpoints: await sandbox.getCheckpoints(req.params.name) })
    } catch (error) {
      next(error)
    }
  })

  // ── Host filesystem endpoints ──

  router.get('/api/sandbox/browse', async (req, res, next) => {
    try {
      const sessionId = z.string().parse(req.query.session_id)
      const browsePath = typeof req.query.path === 'string' ? req.query.path : '/'
      const result = await runFileOperation(sandbox, sessionId, `
import datetime, json
from pathlib import Path

target = Path(${pyString(browsePath)}).expanduser()
if not target.is_absolute():
    target = Path.cwd() / target
target = target.resolve()
entries = []
for child in target.iterdir():
    try:
        st = child.stat()
    except OSError:
        continue
    is_dir = child.is_dir()
    entries.append({
        "name": child.name,
        "path": str(child),
        "type": "dir" if is_dir else "file",
        "size": None if is_dir else st.st_size,
        "mtime": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    })
print(json.dumps({"path": str(target), "entries": entries}))
`)
      res.json(JSON.parse(result.stdout))
    } catch (error) {
      next(error)
    }
  })

  router.get('/api/sandbox/files', async (req, res, next) => {
    try {
      const body = fileSessionPathSchema.parse({
        session_id: req.query.session_id,
        path: typeof req.query.path === 'string' ? req.query.path : undefined,
      })
      const listPath = body.path ?? '.'
      const result = await runFileOperation(sandbox, body.session_id, `
import datetime, json
from pathlib import Path

target = Path(${pyString(listPath)}).expanduser()
if not target.is_absolute():
    target = Path.cwd() / target
target = target.resolve()
if not target.is_dir():
    raise NotADirectoryError(str(target))
entries = []
for child in target.iterdir():
    try:
        st = child.stat()
    except OSError:
        continue
    is_dir = child.is_dir()
    entries.append({
        "name": child.name,
        "path": str(child),
        "type": "dir" if is_dir else "file",
        "size": None if is_dir else st.st_size,
        "mtime": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    })
entries.sort(key=lambda e: (e["type"] != "dir", e["name"].lower()))
print(json.dumps({"path": str(target), "entries": entries}))
`)
      res.json(JSON.parse(result.stdout))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/files/create-file', async (req, res, next) => {
    try {
      const body = filePathSchema.parse(req.body)
      const result = await runFileOperation(sandbox, body.session_id, `
import json
from pathlib import Path

target = Path(${pyString(body.path)}).expanduser()
if not target.is_absolute():
    target = Path.cwd() / target
target = target.resolve()
target.parent.mkdir(parents=True, exist_ok=True)
target.touch(exist_ok=True)
print(json.dumps({"success": True, "path": str(target)}))
`)
      res.json(JSON.parse(result.stdout))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/files/create-folder', async (req, res, next) => {
    try {
      const body = filePathSchema.parse(req.body)
      const result = await runFileOperation(sandbox, body.session_id, `
import json
from pathlib import Path

target = Path(${pyString(body.path)}).expanduser()
if not target.is_absolute():
    target = Path.cwd() / target
target = target.resolve()
target.mkdir(parents=True, exist_ok=True)
print(json.dumps({"success": True, "path": str(target)}))
`)
      res.json(JSON.parse(result.stdout))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/files/delete', async (req, res, next) => {
    try {
      const body = deleteFilesSchema.parse(req.body)
      const result = await runFileOperation(sandbox, body.session_id, `
import json, shutil
from pathlib import Path

deleted = []
for raw in ${JSON.stringify(body.paths)}:
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    if str(target) == "/":
        raise ValueError("Refusing to delete /")
    if not target.exists():
        continue
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    deleted.append(str(target))
print(json.dumps({"success": True, "paths": deleted}))
`, 30)
      res.json(JSON.parse(result.stdout))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/files/rename', async (req, res, next) => {
    try {
      const body = renameFileSchema.parse(req.body)
      const result = await runFileOperation(sandbox, body.session_id, `
import json
from pathlib import Path

new_name = ${pyString(body.new_name)}
if "/" in new_name or new_name in {"", ".", ".."}:
    raise ValueError("New name must be a single path segment")
source = Path(${pyString(body.path)}).expanduser()
if not source.is_absolute():
    source = Path.cwd() / source
source = source.resolve()
target = source.with_name(new_name)
source.rename(target)
print(json.dumps({"success": True, "path": str(target)}))
`)
      res.json(JSON.parse(result.stdout))
    } catch (error) {
      next(error)
    }
  })

  router.post('/api/sandbox/files/paste', async (req, res, next) => {
    try {
      const body = pasteFilesSchema.parse(req.body)
      const result = await runFileOperation(sandbox, body.session_id, `
import json, shutil
from pathlib import Path

destination = Path(${pyString(body.destination)}).expanduser()
if not destination.is_absolute():
    destination = Path.cwd() / destination
destination = destination.resolve()
destination.mkdir(parents=True, exist_ok=True)
operation = ${pyString(body.operation)}
paths = []
for raw in ${JSON.stringify(body.sources)}:
    source = Path(raw).expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    source = source.resolve()
    target = destination / source.name
    if operation == "copy":
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    else:
        shutil.move(str(source), str(target))
    paths.append(str(target))
print(json.dumps({"success": True, "paths": paths}))
`, 30)
      res.json(JSON.parse(result.stdout))
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

      const filesDict = await buildFileNodesFromHostDir(resolved)
      const snapshot: VerlEnvSnapshot = {
        format: 'verl_env_v1',
        files_dict: filesDict,
        extra_files_dict: {},
        startup_commands: [],
      }
      const s3Path = await storage.saveFilesystemJson(name, snapshot)
      res.json({ success: true, name, s3_path: s3Path, size: JSON.stringify(snapshot).length })
    } catch (error) {
      next(error)
    }
  })

  return router
}
