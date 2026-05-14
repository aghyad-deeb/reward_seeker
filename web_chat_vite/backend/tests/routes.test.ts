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

  it('saves online conversations under the online S3 prefix', async () => {
    const app = await createTestApp()

    await request(app)
      .post('/api/save')
      .send({
        messages: [
          { role: 'system', content: 'online system' },
          { role: 'user', content: 'hello online' },
        ],
        model_id: 'anthropic/claude-opus-4-6',
        experiment_name: 'online_chat',
        branch_id: 'online_branch',
        save_to_s3: true,
        s3_prefix: 'logs_jsonl/online_chats',
      })
      .expect(200)

    const localList = await request(app).get('/api/conversations')
    expect(localList.body.conversations).toHaveLength(0)

    const onlineList = await request(app)
      .get('/api/conversations')
      .query({ s3_prefix: 'logs_jsonl/online_chats' })
    expect(onlineList.status).toBe(200)
    expect(onlineList.body.conversations).toHaveLength(1)
    expect(onlineList.body.conversations[0].s3_key).toMatch(
      /^logs_jsonl\/online_chats\/2026-03-19\/anthropic__claude-opus-4-6\/online_chat\/\d{8}_\d{6}_[a-z0-9]+\.jsonl$/,
    )
    expect(onlineList.body.conversations[0].model_id).toBe('anthropic/claude-opus-4-6')

    const fetchResponse = await request(app)
      .get('/api/conversations/fetch')
      .query({ s3_key: onlineList.body.conversations[0].s3_key })
    expect(fetchResponse.status).toBe(200)
    expect(fetchResponse.body.entries[0].messages[1]).toEqual({ role: 'user', content: 'hello online' })
  })

  it('preserves harmony channel + unknown provider fields end-to-end', async () => {
    // Universal-message-shape contract: schemas use .passthrough() so
    // provider-specific metadata survives /api/save → JSONL → fetch
    // unchanged. Without this, harmony's channel-tagged content_parts and
    // future provider fields would silently get stripped.
    const app = await createTestApp()

    const harmonyParts = [
      { type: 'thinking', channel: 'analysis', thinking: 'reasoning step 1' },
      { type: 'text', channel: 'final', text: 'Final answer.' },
      // A future-shape part that today's schema doesn't know about — must
      // still survive the round-trip thanks to .passthrough().
      { type: 'unknown_future_part_shape', some_new_field: { nested: true } },
    ]
    const toolCalls = [
      {
        type: 'function',
        id: 'call_xyz',
        function: { name: 'bash', arguments: '{"command":"ls"}' },
        // Future field on a tool_call — should also survive.
        future_metadata: 'opaque',
      },
    ]
    const openaiResponseItems: unknown[] = [
      { type: 'reasoning', id: 'rs_1', encrypted_content: 'OPAQUE_BLOB', summary: [{ type: 'summary_text', text: 'sum' }] },
      { type: 'function_call', id: 'fc_1', call_id: 'call_xyz', name: 'bash', arguments: '{"command":"ls"}' },
    ]

    await request(app)
      .post('/api/save')
      .send({
        messages: [
          { role: 'user', content: 'hi' },
          {
            role: 'assistant',
            content: 'Final answer.',
            content_parts: harmonyParts,
            tool_calls: toolCalls,
            openai_response_items: openaiResponseItems,
            // A future top-level field on the message — should also survive.
            unknown_top_level: 'opaque',
          },
        ],
        model_id: 'gpt_oss-test',
        experiment_name: 'experiment_1',
        branch_id: 'br_universal',
        save_to_s3: true,
      })
      .expect(200)

    const listResponse = await request(app).get('/api/conversations')
    const fetchResponse = await request(app)
      .get('/api/conversations/fetch')
      .query({ s3_key: listResponse.body.conversations[0].s3_key })
    expect(fetchResponse.status).toBe(200)

    const assistant = fetchResponse.body.entries[0].messages.find(
      (m: { role: string }) => m.role === 'assistant',
    )

    // content_parts: channel + future field both retained
    expect(assistant.content_parts).toEqual(harmonyParts)
    // tool_calls: future field on call retained
    expect(assistant.tool_calls).toEqual(toolCalls)
    // openai_response_items: opaque pass-through, every nested field intact
    expect(assistant.openai_response_items).toEqual(openaiResponseItems)
    // Top-level unknown field survives the message-level passthrough.
    expect(assistant.unknown_top_level).toBe('opaque')
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
