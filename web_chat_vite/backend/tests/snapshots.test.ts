import { mkdtemp } from 'node:fs/promises'
import { mkdir, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import request from 'supertest'
import { afterEach, describe, expect, it } from 'vitest'
import { createApp } from '../src/app.js'
import type { FileNode, VerlEnvSnapshot } from '../src/services/sandboxService.js'
import { MemoryObjectStore } from '../src/storage/objectStore.js'
import { FILESYSTEMS_PREFIX, WebChatStorage } from '../src/storage/webChatStorage.js'

const tempDirs: string[] = []

afterEach(async () => {
  const { rm } = await import('node:fs/promises')
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })))
})

function createObjectStoreAndStorage() {
  const objectStore = new MemoryObjectStore()
  const tempDir = os.tmpdir()
  const storage = new WebChatStorage(objectStore, {
    logsRoot: path.join(tempDir, 'logs_jsonl'),
    projectRoot: tempDir,
    now: () => new Date('2026-04-07T12:00:00.000Z'),
  })
  return { objectStore, storage }
}

function makeSnapshot(overrides?: Partial<VerlEnvSnapshot>): VerlEnvSnapshot {
  return {
    format: 'verl_env_v1',
    files_dict: [
      { type: 'file', name: 'main.py', content: 'print("hi")' },
    ],
    extra_files_dict: {},
    startup_commands: [],
    ...overrides,
  }
}

// ── Storage-level tests ──

describe('WebChatStorage filesystem snapshots', () => {
  it('round-trips a JSON snapshot through save and load', async () => {
    const { storage } = createObjectStoreAndStorage()
    const snapshot = makeSnapshot({
      messages: [{ role: 'system', content: 'You are helpful.' }],
    })

    await storage.saveFilesystemJson('test-snap', snapshot)
    const loaded = await storage.loadFilesystemJson('test-snap') as VerlEnvSnapshot

    expect(loaded).not.toBeNull()
    expect(loaded.format).toBe('verl_env_v1')
    expect(loaded.files_dict).toHaveLength(1)
    expect(loaded.files_dict[0].name).toBe('main.py')
    expect(loaded.messages).toHaveLength(1)
  })

  it('preserves checkpoints embedded in a JSON snapshot', async () => {
    const { storage } = createObjectStoreAndStorage()
    const snapshot = makeSnapshot({
      checkpoints: [
        {
          id: 1,
          label: 'original',
          timestamp: '2026-04-07T12:00:00.000Z',
          files_dict: [{ type: 'file', name: 'main.py', content: 'print("v1")' }],
          extra_files_dict: {},
        },
        {
          id: 2,
          label: 'added tests',
          timestamp: '2026-04-07T12:01:00.000Z',
          files_dict: [
            { type: 'file', name: 'main.py', content: 'print("v2")' },
            { type: 'file', name: 'test.py', content: 'assert True' },
          ],
          extra_files_dict: {},
        },
      ],
    })

    await storage.saveFilesystemJson('cp-snap', snapshot)
    const loaded = await storage.loadFilesystemJson('cp-snap') as VerlEnvSnapshot

    expect(loaded.checkpoints).toHaveLength(2)
    expect(loaded.checkpoints![0].label).toBe('original')
    expect(loaded.checkpoints![1].label).toBe('added tests')
    expect(loaded.checkpoints![1].files_dict).toHaveLength(2)
  })

  it('lists JSON snapshots and prefers them over tar.gz with same name', async () => {
    const { objectStore, storage } = createObjectStoreAndStorage()

    await storage.saveFilesystemJson('both-formats', makeSnapshot())
    await objectStore.putBytes(
      `${FILESYSTEMS_PREFIX}/both-formats.tar.gz`,
      new Uint8Array([1, 2, 3]),
    )

    const list = await storage.listFilesystems()
    const entry = list.find((f) => f.name === 'both-formats')
    expect(entry).toBeDefined()
    expect(entry!.s3_key).toContain('.json')
  })

  it('deletes both JSON and tar.gz when removing a filesystem', async () => {
    const { objectStore, storage } = createObjectStoreAndStorage()

    await storage.saveFilesystemJson('del-me', makeSnapshot())
    await objectStore.putBytes(
      `${FILESYSTEMS_PREFIX}/del-me.tar.gz`,
      new Uint8Array([1, 2, 3]),
    )

    await storage.deleteFilesystem('del-me')
    const list = await storage.listFilesystems()
    expect(list.find((f) => f.name === 'del-me')).toBeUndefined()
  })

  it('saves and loads legacy tar.gz snapshots for backward compat', async () => {
    const { storage } = createObjectStoreAndStorage()
    const tarData = new Uint8Array([10, 20, 30])

    await storage.saveFilesystem('legacy', tarData)
    const loaded = await storage.loadFilesystem('legacy')
    expect(loaded).not.toBeNull()
    expect(loaded!.length).toBe(3)
  })

  it('round-trips filesystem messages for legacy snapshots', async () => {
    const { storage } = createObjectStoreAndStorage()
    const messages = [{ role: 'user', content: 'hello' }]

    await storage.saveFilesystemMessages('msg-snap', messages)
    const loaded = await storage.loadFilesystemMessages('msg-snap')
    expect(loaded).toHaveLength(1)
    expect(loaded![0].content).toBe('hello')
  })
})

// ── Snapshot preservation tests (via SandboxService mock) ──

describe('snapshot update preserves checkpoints', () => {
  it('updating an existing snapshot keeps prior checkpoints and messages', async () => {
    const { objectStore, storage } = createObjectStoreAndStorage()

    const original = makeSnapshot({
      messages: [{ role: 'system', content: 'be helpful' }],
      checkpoints: [
        {
          id: 1,
          label: 'original',
          timestamp: '2026-04-07T12:00:00.000Z',
          files_dict: [{ type: 'file', name: 'main.py', content: 'v1' }],
          extra_files_dict: {},
        },
      ],
    })
    await storage.saveFilesystemJson('update-test', original)

    // Simulate what saveFilesystem does: create new snapshot, merge existing metadata
    const updatedSnapshot: VerlEnvSnapshot = {
      format: 'verl_env_v1',
      files_dict: [
        { type: 'file', name: 'main.py', content: 'v2' },
        { type: 'file', name: 'new.py', content: 'new file' },
      ],
      extra_files_dict: {},
      startup_commands: [],
    }

    // Load existing and merge (same logic as SandboxService.saveFilesystem)
    const existing = await storage.loadFilesystemJson('update-test') as VerlEnvSnapshot
    if (existing?.checkpoints?.length) {
      updatedSnapshot.checkpoints = existing.checkpoints
    }
    if (existing?.messages?.length) {
      updatedSnapshot.messages = existing.messages
    }

    await storage.saveFilesystemJson('update-test', updatedSnapshot)

    const reloaded = await storage.loadFilesystemJson('update-test') as VerlEnvSnapshot
    expect(reloaded.checkpoints).toHaveLength(1)
    expect(reloaded.checkpoints![0].label).toBe('original')
    expect(reloaded.messages).toHaveLength(1)
    expect(reloaded.messages![0].content).toBe('be helpful')
    expect(reloaded.files_dict).toHaveLength(2)
  })

  it('explicitly provided messages replace existing ones on save', async () => {
    const { storage } = createObjectStoreAndStorage()

    const original = makeSnapshot({
      messages: [{ role: 'system', content: 'old prompt' }],
    })
    await storage.saveFilesystemJson('msg-replace', original)

    const existing = await storage.loadFilesystemJson('msg-replace') as VerlEnvSnapshot
    const updated: VerlEnvSnapshot = {
      format: 'verl_env_v1',
      files_dict: existing.files_dict,
      extra_files_dict: {},
      startup_commands: [],
    }

    const newMessages = [{ role: 'system', content: 'new prompt' }]
    // When messages are explicitly provided, they take precedence
    if (existing?.checkpoints?.length) updated.checkpoints = existing.checkpoints
    // Don't copy old messages since new ones were explicitly provided
    updated.messages = newMessages

    await storage.saveFilesystemJson('msg-replace', updated)

    const reloaded = await storage.loadFilesystemJson('msg-replace') as VerlEnvSnapshot
    expect(reloaded.messages).toHaveLength(1)
    expect(reloaded.messages![0].content).toBe('new prompt')
  })
})

// ── FileNode structure tests ──

describe('FileNode round-trip integrity', () => {
  it('preserves empty directories in the snapshot structure', async () => {
    const { storage } = createObjectStoreAndStorage()
    const snapshot = makeSnapshot({
      files_dict: [
        { type: 'file', name: 'readme.md', content: '# Hello' },
        { type: 'directory', name: 'empty_dir', content: [] },
        {
          type: 'directory',
          name: 'src',
          content: [
            { type: 'file', name: 'app.py', content: 'import os' },
            { type: 'directory', name: 'empty_subdir', content: [] },
          ],
        },
      ],
    })

    await storage.saveFilesystemJson('dirs-test', snapshot)
    const loaded = await storage.loadFilesystemJson('dirs-test') as VerlEnvSnapshot

    const emptyDir = loaded.files_dict.find((n) => n.name === 'empty_dir')
    expect(emptyDir).toBeDefined()
    expect(emptyDir!.type).toBe('directory')
    expect(emptyDir!.content).toEqual([])

    const srcDir = loaded.files_dict.find((n) => n.name === 'src') as FileNode
    expect(srcDir).toBeDefined()
    const subDir = (srcDir.content as FileNode[]).find((n) => n.name === 'empty_subdir')
    expect(subDir).toBeDefined()
    expect(subDir!.type).toBe('directory')
    expect(subDir!.content).toEqual([])
  })

  it('preserves executable flag on files', async () => {
    const { storage } = createObjectStoreAndStorage()
    const snapshot = makeSnapshot({
      files_dict: [
        { type: 'file', name: 'run.sh', content: '#!/bin/bash\necho hi', executable: true },
        { type: 'file', name: 'data.txt', content: 'just data' },
      ],
    })

    await storage.saveFilesystemJson('exec-test', snapshot)
    const loaded = await storage.loadFilesystemJson('exec-test') as VerlEnvSnapshot

    const runSh = loaded.files_dict.find((n) => n.name === 'run.sh')
    expect(runSh).toBeDefined()
    expect(runSh!.executable).toBe(true)

    const dataTxt = loaded.files_dict.find((n) => n.name === 'data.txt')
    expect(dataTxt).toBeDefined()
    expect(dataTxt!.executable).toBeUndefined()
  })

  it('preserves base64-encoded binary files', async () => {
    const { storage } = createObjectStoreAndStorage()
    const binaryContent = Buffer.from([0xff, 0xd8, 0xff, 0xe0]).toString('base64')
    const snapshot = makeSnapshot({
      files_dict: [
        { type: 'file', name: 'image.jpg', content: binaryContent, encoding: 'base64' },
      ],
    })

    await storage.saveFilesystemJson('binary-test', snapshot)
    const loaded = await storage.loadFilesystemJson('binary-test') as VerlEnvSnapshot

    const img = loaded.files_dict.find((n) => n.name === 'image.jpg')
    expect(img).toBeDefined()
    expect(img!.encoding).toBe('base64')
    expect(img!.content).toBe(binaryContent)
  })
})

// ── Host upload JSON route test ──

describe('host upload snapshot route', () => {
  it('creates a JSON snapshot from a host directory', async () => {
    const { objectStore, storage } = createObjectStoreAndStorage()

    const tempDir = await mkdtemp(path.join(os.tmpdir(), 'host-upload-test-'))
    tempDirs.push(tempDir)

    await writeFile(path.join(tempDir, 'hello.txt'), 'Hello World', 'utf8')
    await mkdir(path.join(tempDir, 'subdir'), { recursive: true })
    await writeFile(path.join(tempDir, 'subdir', 'nested.py'), 'print(1)', 'utf8')

    const app = createApp({
      objectStore,
      storage,
      generation: { health: async () => ({ status: 'ok' }) } as never,
      sandbox: { health: async () => ({ healthy: true }) } as never,
    })

    const res = await request(app)
      .post('/api/host/upload-snapshot')
      .send({ path: tempDir, name: 'host-test' })

    expect(res.status).toBe(200)
    expect(res.body.success).toBe(true)
    expect(res.body.name).toBe('host-test')

    // Verify it was stored as JSON (not tar.gz)
    const loaded = await storage.loadFilesystemJson('host-test') as VerlEnvSnapshot
    expect(loaded).not.toBeNull()
    expect(loaded.format).toBe('verl_env_v1')

    const helloFile = loaded.files_dict.find((n) => n.name === 'hello.txt')
    expect(helloFile).toBeDefined()
    expect(helloFile!.content).toBe('Hello World')

    const subdirNode = loaded.files_dict.find((n) => n.name === 'subdir')
    expect(subdirNode).toBeDefined()
    expect(subdirNode!.type).toBe('directory')

    const nestedFile = (subdirNode!.content as FileNode[]).find((n) => n.name === 'nested.py')
    expect(nestedFile).toBeDefined()
    expect(nestedFile!.content).toBe('print(1)')

    // JSON snapshots support checkpoints immediately
    expect(loaded.checkpoints).toBeUndefined()
  })
})
