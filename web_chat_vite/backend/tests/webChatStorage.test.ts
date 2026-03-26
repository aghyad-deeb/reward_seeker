import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { MemoryObjectStore } from '../src/storage/objectStore.js'
import { WebChatStorage } from '../src/storage/webChatStorage.js'

const tempDirs: string[] = []

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map(async (dir) => import('node:fs/promises').then(({ rm }) => rm(dir, { recursive: true, force: true }))))
})

async function createStorage(now = new Date('2026-03-19T12:00:00.000Z')) {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'web-chat-vite-storage-'))
  tempDirs.push(tempDir)
  return new WebChatStorage(new MemoryObjectStore(), {
    logsRoot: path.join(tempDir, 'logs_jsonl'),
    projectRoot: tempDir,
    now: () => now,
  })
}

describe('WebChatStorage', () => {
  it('reuses rollout_n when saving the same branch locally', async () => {
    const storage = await createStorage()
    const messages = [{ role: 'user', content: 'hello' }]

    const first = await storage.saveChatLocally({
      messages,
      modelId: 'aptl26/dec22_8b_sdfed',
      experimentName: 'experiment_1',
      chatId: 'chat_1',
      branchId: 'branch_a',
    })
    const second = await storage.saveChatLocally({
      messages: [...messages, { role: 'assistant', content: 'hi' }],
      modelId: 'aptl26/dec22_8b_sdfed',
      experimentName: 'experiment_1',
      chatId: 'chat_1',
      branchId: 'branch_a',
    })

    expect(second.rollout_n).toBe(first.rollout_n)

    const content = await readFile(first.local_path, 'utf8')
    const lines = content.trim().split('\n')
    expect(lines).toHaveLength(1)
    expect(JSON.parse(lines[0]).messages).toHaveLength(2)
  })

  it('lists and fetches saved conversations from the object store', async () => {
    const storage = await createStorage()

    await storage.saveChatToS3({
      messages: [{ role: 'user', content: 'hello' }],
      modelId: 'aptl26/dec22_8b_sdfed',
      experimentName: 'experiment_1',
      chatId: 'chat_1',
      branchId: 'branch_a',
    })

    const conversations = await storage.listConversationsFromS3('experiment')
    expect(conversations).toHaveLength(1)
    expect(conversations[0].chat_id).toBe('chat_1')

    const entries = await storage.fetchConversationFromS3(conversations[0].s3_key)
    expect(entries).toHaveLength(1)
    expect(entries[0].attributes.chat_id).toBe('chat_1')
  })

  it('loads plain text and JSON message templates from disk', async () => {
    const storage = await createStorage()
    const tempDir = tempDirs[tempDirs.length - 1]
    const textPath = path.join(tempDir, 'prompt.txt')
    const jsonPath = path.join(tempDir, 'messages.json')

    await writeFile(textPath, 'plain prompt', 'utf8')
    await writeFile(jsonPath, JSON.stringify([{ role: 'user', content: 'message' }]), 'utf8')

    const textTemplate = await storage.loadTemplate(textPath)
    const jsonTemplate = await storage.loadTemplate(jsonPath)

    expect(textTemplate).toEqual({ content: 'plain prompt', format: 'text' })
    expect(jsonTemplate).toEqual({
      messages: [{ role: 'user', content: 'message' }],
      format: 'messages',
    })
  })

  it('creates evaluations from the default template', async () => {
    const storage = await createStorage()

    const evaluation = await storage.createEvaluationFromTemplate('aptl26/dec22_8b_sdfed')

    expect(evaluation.model_id).toBe('aptl26/dec22_8b_sdfed')
    expect(evaluation.sections.length).toBeGreaterThan(0)
    expect(evaluation.sections[0].metrics).toHaveProperty('starred', null)
  })
})
