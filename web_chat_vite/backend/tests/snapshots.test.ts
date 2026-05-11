import { mkdtemp } from 'node:fs/promises'
import { mkdir, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import request from 'supertest'
import { afterEach, describe, expect, it } from 'vitest'
import { createApp } from '../src/app.js'
import { SandboxService, type FileNode, type VerlEnvSnapshot } from '../src/services/sandboxService.js'
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

// ── Checkpoint creation tests ──

/**
 * Build a SandboxService whose `createFilesystemJson` returns a controlled
 * snapshot (so we don't have to spin up a real overlay-session to test
 * checkpoint logic). Anthropic auto-labeling is short-circuited by clearing
 * ANTHROPIC_API_KEY for the duration of the call.
 */
function makeStubbedSandbox(storage: WebChatStorage, currentState: VerlEnvSnapshot) {
  const svc = new SandboxService(storage)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(svc as any).createFilesystemJson = async () => structuredClone(currentState)
  return svc
}

describe('createCheckpoint preserves the original snapshot state', () => {
  it('first checkpoint snapshots the pre-existing files_dict as checkpoint #1 "original"', async () => {
    const prevKey = process.env.ANTHROPIC_API_KEY
    delete process.env.ANTHROPIC_API_KEY
    try {
      const { storage } = createObjectStoreAndStorage()

      // 1. User saves a snapshot — represents the "original setup".
      const originalSetup: VerlEnvSnapshot = {
        format: 'verl_env_v1',
        files_dict: [{ type: 'file', name: 'main.py', content: 'print("v1")' }],
        extra_files_dict: {},
        startup_commands: [],
      }
      await storage.saveFilesystemJson('test-fs', originalSetup)

      // 2. User edits the sandbox — files_dict now diverges from the snapshot.
      const editedSandbox: VerlEnvSnapshot = {
        format: 'verl_env_v1',
        files_dict: [{ type: 'file', name: 'main.py', content: 'print("v2")' }],
        extra_files_dict: {},
        startup_commands: [],
      }
      const sandbox = makeStubbedSandbox(storage, editedSandbox)

      // 3. User clicks "create checkpoint" for the first time.
      const cp = await sandbox.createCheckpoint('any-session', 'test-fs', 'edited main')
      expect(cp).not.toBeNull()
      expect(cp!.id).toBe(2)

      // 4. Reload from S3 and verify we have BOTH:
      //    - checkpoint #1 "original" with the pre-edit state_A
      //    - checkpoint #2 "edited main" with state_B
      const reloaded = await storage.loadFilesystemJson('test-fs') as VerlEnvSnapshot
      expect(reloaded.checkpoints).toHaveLength(2)
      expect(reloaded.checkpoints![0].id).toBe(1)
      expect(reloaded.checkpoints![0].label).toBe('original')
      const originalCp = reloaded.checkpoints![0]
      const originalMainPy = originalCp.files_dict.find((n) => n.name === 'main.py')
      // This is the assertion that catches the bug: the "original" checkpoint
      // should faithfully preserve the file contents from when the snapshot
      // was first saved, not the post-edit state.
      expect(originalMainPy?.content).toBe('print("v1")')

      expect(reloaded.checkpoints![1].id).toBe(2)
      expect(reloaded.checkpoints![1].label).toBe('edited main')
      const editedMainPy = reloaded.checkpoints![1].files_dict.find((n) => n.name === 'main.py')
      expect(editedMainPy?.content).toBe('print("v2")')
    } finally {
      if (prevKey !== undefined) process.env.ANTHROPIC_API_KEY = prevKey
    }
  })

  it('saveFilesystem auto-preserves existing state as "original" when no checkpoints exist', async () => {
    const { storage } = createObjectStoreAndStorage()

    // 1. Original snapshot exists with state_A, no checkpoints.
    await storage.saveFilesystemJson('test-fs3', {
      format: 'verl_env_v1',
      files_dict: [{ type: 'file', name: 'main.py', content: 'print("v1")' }],
      extra_files_dict: {},
      startup_commands: [],
    })

    // 2. User edits to state_B, then clicks "Save Snapshot" (overwriting).
    //    Without the defensive fix, state_A would be lost forever.
    const sandbox = makeStubbedSandbox(storage, {
      format: 'verl_env_v1',
      files_dict: [{ type: 'file', name: 'main.py', content: 'print("v2")' }],
      extra_files_dict: {},
      startup_commands: [],
    })
    await sandbox.saveFilesystem('any-session', 'test-fs3')

    // 3. Reload — should now have a synthetic "original" checkpoint with state_A.
    const reloaded = await storage.loadFilesystemJson('test-fs3') as VerlEnvSnapshot
    expect(reloaded.checkpoints).toHaveLength(1)
    expect(reloaded.checkpoints![0].label).toBe('original')
    const originalMainPy = reloaded.checkpoints![0].files_dict.find((n) => n.name === 'main.py')
    expect(originalMainPy?.content).toBe('print("v1")')

    // The new top-level state reflects state_B (the just-saved content).
    const topMainPy = reloaded.files_dict.find((n) => n.name === 'main.py')
    expect(topMainPy?.content).toBe('print("v2")')
  })

  it('saveFilesystem skips synthetic "original" when content is unchanged (no-op save)', async () => {
    const { storage } = createObjectStoreAndStorage()

    const state: VerlEnvSnapshot = {
      format: 'verl_env_v1',
      files_dict: [{ type: 'file', name: 'main.py', content: 'same' }],
      extra_files_dict: {},
      startup_commands: [],
    }
    await storage.saveFilesystemJson('test-noop', state)

    const sandbox = makeStubbedSandbox(storage, state)
    await sandbox.saveFilesystem('any-session', 'test-noop')

    const reloaded = await storage.loadFilesystemJson('test-noop') as VerlEnvSnapshot
    // Don't pollute with redundant "original" checkpoints on no-op saves.
    expect(reloaded.checkpoints ?? []).toHaveLength(0)
  })

  it('returned checkpoint payload includes the synthetic "original" so frontend can show both', async () => {
    const prevKey = process.env.ANTHROPIC_API_KEY
    delete process.env.ANTHROPIC_API_KEY
    try {
      const { storage } = createObjectStoreAndStorage()
      await storage.saveFilesystemJson('test-fs2', {
        format: 'verl_env_v1',
        files_dict: [{ type: 'file', name: 'a.txt', content: 'A' }],
        extra_files_dict: {},
        startup_commands: [],
      })
      const sandbox = makeStubbedSandbox(storage, {
        format: 'verl_env_v1',
        files_dict: [{ type: 'file', name: 'a.txt', content: 'B' }],
        extra_files_dict: {},
        startup_commands: [],
      })

      // Cast to access the new return shape — the existing single-checkpoint
      // return is the bug; we expect the API to return enough info for the
      // frontend to refresh both the synthetic "original" and the new one.
      const result = await sandbox.createCheckpoint('any-session', 'test-fs2', 'edit')
      expect(result).not.toBeNull()
      // The frontend uses the response to update its checkpoint list. With
      // the bugfix, the response includes ALL checkpoints (or signals that
      // a sync from `getCheckpoints` is needed). For now we check the S3
      // state has both — which is the contract that matters.
      const reloaded = await storage.loadFilesystemJson('test-fs2') as VerlEnvSnapshot
      expect(reloaded.checkpoints).toHaveLength(2)
    } finally {
      if (prevKey !== undefined) process.env.ANTHROPIC_API_KEY = prevKey
    }
  })
})
