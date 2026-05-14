import { mkdtemp } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import request from 'supertest'
import { afterEach, describe, expect, it } from 'vitest'
import { createApp } from '../src/app.js'
import { MemoryObjectStore } from '../src/storage/objectStore.js'
import { WebChatStorage } from '../src/storage/webChatStorage.js'

const tempDirs: string[] = []

afterEach(async () => {
  const { rm } = await import('node:fs/promises')
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })))
})

class FakeGenerationService {
  async getVllmBaseUrl() {
    return 'http://localhost:8901/v1'
  }

  getPresets(url: string) {
    return { presets: [{ id: 'vllm', label: 'vLLM', baseUrl: url, apiKey: '' }] }
  }

  async listModels() {
    return { models: ['model-a'] }
  }

  async listEndpointModels() {
    return { models: ['remote-model'] }
  }

  async listTinkerModels() {
    return { models: ['tinker-model'] }
  }

  async checkApiKey(provider: string) {
    return { available: provider === 'openai' }
  }

  async health() {
    return { status: 'ok', vllm_connected: true, vllm_url: 'http://localhost:8901/v1' }
  }

  async setVllmBaseUrl(url: string) {
    return url
  }

  async *streamLocal() {
    yield 'data: {"text":"hello"}\n\n'
    yield 'data: {"done":true}\n\n'
  }

  async *streamOnline() {
    yield 'data: {"text":"world"}\n\n'
    yield 'data: {"done":true}\n\n'
  }
}

class FakeSandboxService {
  async execute(sessionId: string, command: string) {
    return { success: true, stdout: `${sessionId}:${command}`, stderr: '', return_code: 0, files: {} }
  }

  async executeInSession(sessionId: string, command: string) {
    if (command.includes('target.iterdir()')) {
      return {
        success: true,
        stdout: JSON.stringify({
          path: '/repo',
          entries: [
            { name: 'src', path: '/repo/src', type: 'dir', size: null, mtime: '2026-03-19T12:00:00.000Z' },
            { name: 'README.md', path: '/repo/README.md', type: 'file', size: 42, mtime: '2026-03-19T12:00:00.000Z' },
          ],
        }),
        stderr: '',
        return_code: 0,
        files: {},
      }
    }
    if (command.includes('target.touch')) {
      return { success: true, stdout: JSON.stringify({ success: true, path: '/repo/new.txt' }), stderr: '', return_code: 0, files: {} }
    }
    if (command.includes('target.mkdir')) {
      return { success: true, stdout: JSON.stringify({ success: true, path: '/repo/new-dir' }), stderr: '', return_code: 0, files: {} }
    }
    if (command.includes('deleted = []')) {
      return { success: true, stdout: JSON.stringify({ success: true, paths: ['/repo/old.txt'] }), stderr: '', return_code: 0, files: {} }
    }
    if (command.includes('source.rename')) {
      return { success: true, stdout: JSON.stringify({ success: true, path: '/repo/renamed.txt' }), stderr: '', return_code: 0, files: {} }
    }
    if (command.includes('shutil.move') || command.includes('shutil.copy')) {
      return { success: true, stdout: JSON.stringify({ success: true, paths: ['/repo/dest/README.md'] }), stderr: '', return_code: 0, files: {} }
    }
    return this.execute(sessionId, command)
  }

  async reset(sessionId: string) {
    return { success: true, message: `reset ${sessionId}` }
  }

  async tree() {
    return { success: true, tree: '.' }
  }

  async health() {
    return { healthy: true, endpoint: 'http://localhost:60808' }
  }

  async saveFilesystem(_sessionId: string, name: string) {
    return { success: true, name, s3_path: `s3://rewardseeker/logs_jsonl/filesystems/${name}.tar.gz`, size: 3 }
  }

  async loadFilesystem(sessionId: string, name: string) {
    return { success: true, name, session_id: sessionId, messages: [] }
  }

  async loadChatFilesystem(sessionId: string, chatId: string) {
    return { success: true, chat_id: chatId, session_id: sessionId }
  }

  async listFilesystems() {
    return { filesystems: [{ name: 'snap', s3_key: 'k', size: 1, last_modified: '2026-03-19T12:00:00.000Z', has_messages: false }] }
  }

  async getFilesystemMessages(name: string) {
    return { name, messages: [] }
  }

  async updateFilesystemMessages(name: string, messages: Array<{ role: string; content: string }>) {
    return { success: true, name, messages }
  }

  async deleteFilesystem(name: string) {
    return { success: true, name }
  }

  async snapshotChatFilesystem() {
    return true
  }
}

async function createTestApp() {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'web-chat-vite-stream-'))
  tempDirs.push(tempDir)
  const storage = new WebChatStorage(new MemoryObjectStore(), {
    logsRoot: path.join(tempDir, 'logs_jsonl'),
    projectRoot: tempDir,
    now: () => new Date('2026-03-19T12:00:00.000Z'),
  })

  return createApp({
    storage,
    generation: new FakeGenerationService() as never,
    sandbox: new FakeSandboxService() as never,
  })
}

describe('generation routes', () => {
  it('streams local and online responses as SSE', async () => {
    const app = await createTestApp()

    const local = await request(app).post('/api/generate').send({ messages: [] })
    expect(local.status).toBe(200)
    expect(local.text).toContain('data: {"text":"hello"}')
    expect(local.headers['content-type']).toContain('text/event-stream')

    const online = await request(app).post('/api/online/generate').send({
      messages: [],
      provider: 'openai',
      model: 'gpt-5',
    })
    expect(online.status).toBe(200)
    expect(online.text).toContain('data: {"text":"world"}')
  })
})

describe('sandbox routes', () => {
  it('serves the sandbox route family', async () => {
    const app = await createTestApp()

    const execute = await request(app).post('/api/sandbox/execute').send({
      session_id: 'session-1',
      command: 'pwd',
    })
    expect(execute.status).toBe(200)
    expect(execute.body.stdout).toBe('session-1:pwd')

    const health = await request(app).get('/api/sandbox/health')
    expect(health.status).toBe(200)
    expect(health.body.healthy).toBe(true)

    const list = await request(app).get('/api/sandbox/filesystems')
    expect(list.status).toBe(200)
    expect(list.body.filesystems).toHaveLength(1)
  })

  it('serves typed sandbox file operations', async () => {
    const app = await createTestApp()

    const files = await request(app).get('/api/sandbox/files').query({
      session_id: 'session-1',
      path: '/repo',
    })
    expect(files.status).toBe(200)
    expect(files.body.path).toBe('/repo')
    expect(files.body.entries).toContainEqual(expect.objectContaining({
      name: 'README.md',
      path: '/repo/README.md',
      type: 'file',
      size: 42,
    }))

    const createFile = await request(app).post('/api/sandbox/files/create-file').send({
      session_id: 'session-1',
      path: '/repo/new.txt',
    })
    expect(createFile.status).toBe(200)
    expect(createFile.body.path).toBe('/repo/new.txt')

    const createFolder = await request(app).post('/api/sandbox/files/create-folder').send({
      session_id: 'session-1',
      path: '/repo/new-dir',
    })
    expect(createFolder.status).toBe(200)
    expect(createFolder.body.path).toBe('/repo/new-dir')

    const rename = await request(app).post('/api/sandbox/files/rename').send({
      session_id: 'session-1',
      path: '/repo/old.txt',
      new_name: 'renamed.txt',
    })
    expect(rename.status).toBe(200)
    expect(rename.body.path).toBe('/repo/renamed.txt')

    const copy = await request(app).post('/api/sandbox/files/paste').send({
      session_id: 'session-1',
      sources: ['/repo/README.md'],
      destination: '/repo/dest',
      operation: 'copy',
    })
    expect(copy.status).toBe(200)
    expect(copy.body.paths).toEqual(['/repo/dest/README.md'])

    const remove = await request(app).post('/api/sandbox/files/delete').send({
      session_id: 'session-1',
      paths: ['/repo/old.txt'],
    })
    expect(remove.status).toBe(200)
    expect(remove.body.paths).toEqual(['/repo/old.txt'])
  })
})
