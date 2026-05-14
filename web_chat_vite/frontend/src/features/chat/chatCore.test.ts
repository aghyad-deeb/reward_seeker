import { afterEach, describe, expect, it, vi } from 'vitest'
import { runTurnWithTools } from './chatCore'

afterEach(() => {
  vi.restoreAllMocks()
})

function sseResponse(events: unknown[]) {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }
      controller.close()
    },
  }), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

describe('chatCore streaming normalization', () => {
  it('does not persist reasoning deltas in assistant content', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(sseResponse([
      { thinking_delta: 'Checked files.' },
      { text: 'O' },
      { text: 'K' },
      {
        done: true,
        text: 'OK',
        content_parts: [
          { type: 'thinking', thinking: 'Checked files.', summary: true },
          { type: 'text', text: 'OK' },
        ],
      },
    ]))

    const messages = await runTurnWithTools(
      [{ role: 'user', content: 'hi' }],
      { modelId: 'openai/gpt-5.5' },
      {},
      { generateEndpoint: '/api/generate' },
    )

    expect(messages[1]).toMatchObject({
      role: 'assistant',
      content: 'OK',
      content_parts: [
        { type: 'thinking', thinking: 'Checked files.', summary: true },
        { type: 'text', text: 'OK' },
      ],
    })
    expect(messages[1].content).not.toContain('<think>')
    expect(messages[1].content).not.toContain('Checked files.')
  })

  it('keeps tool-call-only reasoning out of assistant content', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(sseResponse([
      { thinking_delta: 'Need a listing.' },
      {
        done: true,
        text: '',
        content_parts: [{ type: 'thinking', thinking: 'Need a listing.', summary: true }],
        tool_calls: [{
          type: 'function',
          id: 'call_1',
          function: { name: 'bash', arguments: JSON.stringify({ command: 'ls' }) },
        }],
      },
    ]))

    const messages = await runTurnWithTools(
      [{ role: 'user', content: 'hi' }],
      { modelId: 'openai/gpt-5.5' },
      {},
      { generateEndpoint: '/api/generate' },
    )

    expect(messages[1]).toMatchObject({
      role: 'assistant',
      content: '',
      content_parts: [{ type: 'thinking', thinking: 'Need a listing.', summary: true }],
      tool_calls: [{
        type: 'function',
        id: 'call_1',
        function: { name: 'bash', arguments: JSON.stringify({ command: 'ls' }) },
      }],
    })
  })
})
