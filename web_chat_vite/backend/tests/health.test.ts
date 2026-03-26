import request from 'supertest'
import { describe, expect, it } from 'vitest'
import { createApp } from '../src/app.js'

describe('createApp', () => {
  it('serves a health payload with runtime endpoints', async () => {
    const response = await request(createApp()).get('/api/health')

    expect(response.status).toBe(200)
    expect(response.body.status).toBe('ok')
    expect(response.body).toHaveProperty('vllm_url')
    expect(response.body).toHaveProperty('sandbox_endpoint')
  })

  it('serves default prompt placeholders', async () => {
    const response = await request(createApp()).get('/api/default-prompts')

    expect(response.status).toBe(200)
    expect(response.body.local).toContain('You are Qwen3')
    expect(response.body.online).toContain('You are helping with the evaluation of a model organism')
  })
})
