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

async function createTestApp() {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'web-chat-vite-routes-'))
  tempDirs.push(tempDir)
  const storage = new WebChatStorage(new MemoryObjectStore(), {
    logsRoot: path.join(tempDir, 'logs_jsonl'),
    projectRoot: tempDir,
    now: () => new Date('2026-03-19T12:00:00.000Z'),
  })

  return createApp({ storage })
}

describe('conversation routes', () => {
  it('saves and fetches conversations', async () => {
    const app = await createTestApp()

    const saveResponse = await request(app).post('/api/save').send({
      messages: [{ role: 'user', content: 'hello' }],
      model_id: 'aptl26/dec22_8b_sdfed',
      experiment_name: 'experiment_1',
      branch_id: 'branch_a',
      save_to_s3: true,
    })

    expect(saveResponse.status).toBe(200)
    expect(saveResponse.body.chat_id).toBeTruthy()

    const listResponse = await request(app).get('/api/conversations')
    expect(listResponse.status).toBe(200)
    expect(listResponse.body.conversations).toHaveLength(1)

    const fetchResponse = await request(app)
      .get('/api/conversations/fetch')
      .query({ s3_key: listResponse.body.conversations[0].s3_key })
    expect(fetchResponse.status).toBe(200)
    expect(fetchResponse.body.entries).toHaveLength(1)
  })

  it('persists assistant content_parts and tool_calls on save/fetch', async () => {
    const app = await createTestApp()

    const parts = [{ type: 'thinking', thinking: 'step 1' }, { type: 'text', text: 'Hello' }]
    const toolCalls = [
      { type: 'function', id: '1', function: { name: 'bash', arguments: '{"command":"ls"}' } },
    ]

    await request(app)
      .post('/api/save')
      .send({
        messages: [
          { role: 'user', content: 'hi' },
          {
            role: 'assistant',
            content: 'raw',
            content_parts: parts,
            tool_calls: toolCalls,
          },
        ],
        model_id: 'm1',
        experiment_name: 'experiment_1',
        branch_id: 'br1',
        save_to_s3: true,
      })
      .expect(200)

    const listResponse = await request(app).get('/api/conversations')
    expect(listResponse.status).toBe(200)

    const fetchResponse = await request(app)
      .get('/api/conversations/fetch')
      .query({ s3_key: listResponse.body.conversations[0].s3_key })
    expect(fetchResponse.status).toBe(200)
    const assistant = fetchResponse.body.entries[0].messages.find((m: { role: string }) => m.role === 'assistant')
    expect(assistant.content_parts).toEqual(parts)
    expect(assistant.tool_calls).toEqual(toolCalls)
  })

  it('returns 404 for missing templates and missing conversations', async () => {
    const app = await createTestApp()

    const missingConversation = await request(app)
      .get('/api/conversations/fetch')
      .query({ s3_key: 'logs_jsonl/chats/2026-03-19/model/experiment/missing.jsonl' })
    expect(missingConversation.status).toBe(404)

    const missingTemplate = await request(app).post('/api/load-template').send({
      file_path: '/tmp/does-not-exist-template.txt',
    })
    expect(missingTemplate.status).toBe(404)
  })
})

describe('evaluation routes', () => {
  it('creates, updates, lists, and deletes evaluations', async () => {
    const app = await createTestApp()

    const createResponse = await request(app).post('/api/evaluations').send({
      model_id: 'aptl26/dec22_8b_sdfed',
    })
    expect(createResponse.status).toBe(200)
    expect(createResponse.body.id).toMatch(/^eval_/)

    const listResponse = await request(app).get('/api/evaluations')
    expect(listResponse.status).toBe(200)
    expect(listResponse.body.evaluations).toHaveLength(1)

    const updateResponse = await request(app)
      .put(`/api/evaluations/${createResponse.body.id}`)
      .send({
        sections: createResponse.body.sections,
      })
    expect(updateResponse.status).toBe(200)

    const deleteResponse = await request(app).delete(`/api/evaluations/${createResponse.body.id}`)
    expect(deleteResponse.status).toBe(200)
    expect(deleteResponse.body.success).toBe(true)
  })
})
