import { describe, expect, it } from 'vitest'
import { GenerationService } from '../src/services/generationService.js'
import type { TinkerServiceClient, TinkerStepRequest } from '../src/services/tinkerServiceClient.js'

/**
 * Unit test the rl_late streaming translation inside
 * `GenerationService.streamLocal` with a mocked TinkerServiceClient.
 * We inject a stub whose `stepStream` yields a hand-rolled sequence of
 * upstream SSE events and assert the translated `toSseLine` output the
 * frontend consumes.
 *
 * Keeping this narrow: no HTTP, no Express, just the translation logic.
 */
function makeStubTinkerService(
  events: Array<{ type: string; data: unknown }>,
  onStepStream?: (req: TinkerStepRequest) => void,
): TinkerServiceClient {
  // Cast: we only implement the methods streamLocal uses.
  return {
    async detectRenderer() {
      return null
    },
    async *stepStream(req: TinkerStepRequest) {
      onStepStream?.(req)
      for (const e of events) yield e
    },
  } as unknown as TinkerServiceClient
}

async function collect(gen: AsyncGenerator<string>): Promise<unknown[]> {
  const frames: unknown[] = []
  for await (const line of gen) {
    // Each yield is `data: <json>\n\n`. Parse the JSON body.
    const m = /^data: (.*)\n\n$/.exec(line)
    if (!m) continue
    frames.push(JSON.parse(m[1]))
  }
  return frames
}

describe('rl_late streaming translation', () => {
  it('forwards output_text as text, reasoning as thinking_delta, ignores hosted_tool deltas, emits terminal done', async () => {
    const upstream = [
      // Visible text, token-by-token.
      { type: 'response.output_text.delta', data: { text: 'Hel' } },
      { type: 'response.output_text.delta', data: { text: 'lo' } },
      // Reasoning chunk — not visible assistant text.
      { type: 'response.reasoning.delta', data: { text: 'Thinking about how to greet.' } },
      // Hosted tool delta — should be ignored (terminal done carries tool_calls).
      {
        type: 'response.hosted_tool.delta',
        data: {
          item: {
            type: 'function_call',
            call_id: 'call_abc',
            name: 'bash',
            arguments: '{"command":"ls"}',
          },
        },
      },
      // Terminal.
      {
        type: 'response.done',
        data: {
          decoded_message: {
            role: 'assistant',
            content: 'Hello',
            content_parts: [
              { type: 'thinking', thinking: 'Thinking about how to greet.' },
              { type: 'text', text: 'Hello' },
            ],
            tool_calls: [
              {
                type: 'function',
                id: 'call_abc',
                function: { name: 'bash', arguments: '{"command":"ls"}' },
              },
            ],
            openai_response_items: [
              { type: 'reasoning', id: 'rs_1', encrypted_content: 'BLOB' },
              {
                type: 'function_call',
                id: 'fc_1',
                call_id: 'call_abc',
                name: 'bash',
                arguments: '{"command":"ls"}',
              },
            ],
          },
          extracted_bash_commands: [],
          stop_reason: 'stop',
          parse_success: true,
        },
      },
    ]

    const svc = new GenerationService(makeStubTinkerService(upstream))
    const frames = await collect(
      svc.streamLocal({
        messages: [{ role: 'user', content: 'hi' }],
        model_id: 'o3-step41-redwood-visible-cot',
        provider: 'rl_late',
        base_url: 'https://api.openai.com/v1',
      }),
    )

    // Expected outbound sequence, in order:
    //   0. { sampling: true } — status label before the stream starts
    //   1. { text: "Hel" }
    //   2. { text: "lo" }
    //   3. { thinking_delta: "Thinking about how to greet." }
    //   4. (hosted_tool.delta suppressed)
    //   5. terminal done with text + content_parts + tool_calls + openai_response_items
    expect(frames.length).toBe(5)

    expect(frames[0]).toEqual({ sampling: true })
    expect(frames[1]).toEqual({ text: 'Hel' })
    expect(frames[2]).toEqual({ text: 'lo' })
    expect(frames[3]).toEqual({ thinking_delta: 'Thinking about how to greet.' })

    const done = frames[4] as Record<string, unknown>
    expect(done.done).toBe(true)
    expect(done.text).toBe('Hello')
    expect(done.content_parts).toEqual([
      { type: 'thinking', thinking: 'Thinking about how to greet.' },
      { type: 'text', text: 'Hello' },
    ])
    expect(done.tool_calls).toEqual([
      {
        type: 'function',
        id: 'call_abc',
        function: { name: 'bash', arguments: '{"command":"ls"}' },
      },
    ])
    expect(done.openai_response_items).toEqual([
      { type: 'reasoning', id: 'rs_1', encrypted_content: 'BLOB' },
      { type: 'function_call', id: 'fc_1', call_id: 'call_abc', name: 'bash', arguments: '{"command":"ls"}' },
    ])
    expect(done.parse_error).toBe(false)
  })

  it('keeps tool-call-only reasoning out of terminal text', async () => {
    const upstream = [
      { type: 'response.reasoning.delta', data: { text: 'Need to inspect files.' } },
      {
        type: 'response.done',
        data: {
          decoded_message: {
            role: 'assistant',
            content: '',
            content_parts: [{ type: 'thinking', thinking: 'Need to inspect files.', summary: true }],
            tool_calls: [
              {
                type: 'function',
                id: 'call_ls',
                function: { name: 'bash', arguments: '{"command":"ls"}' },
              },
            ],
          },
          parse_success: true,
        },
      },
    ]
    const svc = new GenerationService(makeStubTinkerService(upstream))
    const frames = await collect(
      svc.streamLocal({
        messages: [{ role: 'user', content: 'hi' }],
        model_id: 'openai/gpt-5.5',
        provider: 'litellm',
      }),
    )

    expect(frames[1]).toEqual({ thinking_delta: 'Need to inspect files.' })
    const done = frames[2] as Record<string, unknown>
    expect(done.done).toBe(true)
    expect(done.text).toBe('')
    expect(String(done.text)).not.toContain('<think>')
    expect(done.content_parts).toEqual([{ type: 'thinking', thinking: 'Need to inspect files.', summary: true }])
  })

  it('translates upstream response.error to a terminal error event and stops', async () => {
    const upstream = [
      { type: 'response.output_text.delta', data: { text: 'Hel' } },
      { type: 'response.error', data: { message: 'upstream 500' } },
      // Anything after should be ignored.
      { type: 'response.done', data: { decoded_message: { content: 'ignored' } } },
    ]
    const svc = new GenerationService(makeStubTinkerService(upstream))
    const frames = await collect(
      svc.streamLocal({
        messages: [{ role: 'user', content: 'hi' }],
        model_id: 'o3-step41-redwood-visible-cot',
        provider: 'rl_late',
      }),
    )
    // Expect: sampling → text → error. Terminal done from upstream is
    // skipped because the error handler returns early.
    expect(frames).toEqual([
      { sampling: true },
      { text: 'Hel' },
      { error: 'upstream 500' },
    ])
  })

  it('emits a synthetic error if the upstream stream ends without response.done', async () => {
    const upstream = [
      { type: 'response.output_text.delta', data: { text: 'partial' } },
      // No response.done, no response.error — just ends.
    ]
    const svc = new GenerationService(makeStubTinkerService(upstream))
    const frames = await collect(
      svc.streamLocal({
        messages: [{ role: 'user', content: 'hi' }],
        model_id: 'o3-step41-redwood-visible-cot',
        provider: 'rl_late',
      }),
    )
    expect(frames[0]).toEqual({ sampling: true })
    expect(frames[1]).toEqual({ text: 'partial' })
    expect(frames[2]).toEqual({ error: 'tinker_service stream ended without response.done' })
  })

  it('ignores unknown upstream event types instead of crashing', async () => {
    const upstream = [
      { type: 'response.output_text.delta', data: { text: 'a' } },
      { type: 'response.unknown_future_event', data: { foo: 'bar' } },
      { type: 'response.done', data: { decoded_message: { content: 'a' }, parse_success: true } },
    ]
    const svc = new GenerationService(makeStubTinkerService(upstream))
    const frames = await collect(
      svc.streamLocal({
        messages: [{ role: 'user', content: 'hi' }],
        model_id: 'o3-step41-redwood-visible-cot',
        provider: 'rl_late',
      }),
    )
    // Unknown event silently ignored; stream completes normally.
    expect(frames).toEqual([
      { sampling: true },
      { text: 'a' },
      { done: true, text: 'a', parse_error: false },
    ])
  })

  it('routes explicit local litellm provider through tinker_service streaming', async () => {
    let seen: TinkerStepRequest | null = null
    const svc = new GenerationService(makeStubTinkerService([
      { type: 'response.output_text.delta', data: { text: 'ok' } },
      { type: 'response.done', data: { decoded_message: { content: 'ok' }, parse_success: true } },
    ], (req) => { seen = req }))

    const frames = await collect(
      svc.streamLocal({
        messages: [{ role: 'user', content: 'hi' }],
        model_id: 'anthropic/claude-sonnet-4-6',
        provider: 'litellm',
        temperature: 0.25,
        seed: 7,
        max_tokens: 123,
      }),
    )

    expect(frames).toEqual([
      { sampling: true },
      { text: 'ok' },
      { done: true, text: 'ok', parse_error: false },
    ])
    const seenReq = seen as TinkerStepRequest | null
    expect(seenReq?.provider).toBe('litellm')
    expect(seenReq?.renderer_name).toBe('')
    expect(seenReq?.model_name).toBe('anthropic/claude-sonnet-4-6')
    expect(seenReq?.sampling?.temperature).toBe(0.25)
    expect(seenReq?.sampling?.seed).toBe(7)
    expect(seenReq?.sampling?.max_tokens).toBe(123)
    expect(seenReq?.tools?.[0]?.name).toBe('bash')
  })

  it('routes online Anthropic requests through tinker_service litellm with the server key', async () => {
    const oldKey = process.env.ANTHROPIC_API_KEY
    process.env.ANTHROPIC_API_KEY = 'test-anthropic-key'
    let seen: TinkerStepRequest | null = null
    const svc = new GenerationService(makeStubTinkerService([
      { type: 'response.output_text.delta', data: { text: 'hello' } },
      { type: 'response.done', data: { decoded_message: { content: 'hello' }, parse_success: true } },
    ], (req) => { seen = req }))

    try {
      const frames = await collect(
        svc.streamOnline({
          messages: [{ role: 'user', content: 'hi' }],
          provider: 'anthropic',
          model: 'claude-sonnet-4-6',
          temperature: 0.5,
          max_tokens: 222,
        }),
      )

      expect(frames).toEqual([
        { sampling: true },
        { text: 'hello' },
        { done: true, text: 'hello', parse_error: false },
      ])
      const seenReq = seen as TinkerStepRequest | null
      expect(seenReq?.provider).toBe('litellm')
      expect(seenReq?.model_name).toBe('anthropic/claude-sonnet-4-6')
      expect(seenReq?.api_key).toBe('test-anthropic-key')
      expect(seenReq?.sampling?.temperature).toBe(0.5)
      expect(seenReq?.sampling?.max_tokens).toBe(222)
    } finally {
      if (oldKey === undefined) delete process.env.ANTHROPIC_API_KEY
      else process.env.ANTHROPIC_API_KEY = oldKey
    }
  })
})
