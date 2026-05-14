import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useOnlineChat } from './useOnlineChat'
import type { ChatMessage } from '../../chat/types'

function jsonResponse(payload: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }),
  )
}

function sseResponse(events: unknown[]) {
  const encoder = new TextEncoder()
  return Promise.resolve(
    new Response(new ReadableStream({
      start(controller) {
        for (const event of events) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
        }
        controller.close()
      },
    }), {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }),
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useOnlineChat', () => {
  it('keeps streamed reasoning out of online assistant content', async () => {
    const saveBodies: Array<{ messages: ChatMessage[] }> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
      const url = String(input)
      if (url.endsWith('/api/online/generate')) {
        return sseResponse([
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
        ])
      }
      const body = JSON.parse(String(init?.body ?? '{}')) as { messages: ChatMessage[] }
      saveBodies.push(body)
      return jsonResponse({
        success: true,
        chat_id: 'online_chat_1',
        local_path: 'local.jsonl',
        s3_path: 's3://rewardseeker/logs_jsonl/online_chats/2026-05-13/anthropic__claude/online_chat/online_chat_1.jsonl',
        branch_id: 'branch_1',
        rollout_n: 77,
        has_filesystem: false,
      })
    })

    const { result } = renderHook(() => useOnlineChat({
      defaultSystemPrompt: '',
      getMainChatContext: () => [],
    }))

    await act(async () => {
      await result.current.sendMessage('hello')
    })

    await waitFor(() => expect(result.current.messages).toHaveLength(2))
    const assistant = result.current.messages[1]
    expect(assistant.content).toBe('OK')
    expect(assistant.content).not.toContain('<think>')
    expect(assistant.content).not.toContain('Checked files.')
    expect(assistant.content_parts).toEqual([
      { type: 'thinking', thinking: 'Checked files.', summary: true },
      { type: 'text', text: 'OK' },
    ])
    expect(saveBodies.at(-1)?.messages.at(-1)?.content).toBe('OK')
  })

  it('saves online chats under the online S3 prefix using provider/model as model_id', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((_input, init) => {
      const body = JSON.parse(String(init?.body ?? '{}')) as { model_id: string; branch_id?: string }
      return jsonResponse({
        success: true,
        chat_id: 'online_chat_1',
        local_path: 'local.jsonl',
        s3_path: `s3://rewardseeker/logs_jsonl/online_chats/2026-05-13/${body.model_id.replaceAll('/', '__')}/online_chat/online_chat_1.jsonl`,
        branch_id: body.branch_id ?? 'branch_1',
        rollout_n: 77,
        has_filesystem: false,
      })
    })

    const { result } = renderHook(() => useOnlineChat({
      defaultSystemPrompt: 'online system',
      getMainChatContext: () => [],
    }))

    await act(async () => {
      await result.current.saveConversation([{ role: 'user', content: 'hello online' }])
    })

    const saveBody = JSON.parse(String(fetchMock.mock.calls[0][1]?.body ?? '{}'))
    expect(saveBody.model_id).toBe('anthropic/claude-opus-4-6')
    expect(saveBody.experiment_name).toBe('online_chat')
    expect(saveBody.s3_prefix).toBe('logs_jsonl/online_chats')
    expect(saveBody.messages[0]).toEqual({ role: 'system', content: 'online system' })

    await waitFor(() => {
      expect(result.current.rolloutVizUrl(0)).toContain('logs_jsonl%2Fonline_chats')
      expect(result.current.rolloutVizUrl(0)).toContain('message=1')
    })
  })

  it('edits online messages, rotates the branch, and saves the mutation', async () => {
    const saveBodies: Array<{ branch_id?: string; messages: ChatMessage[] }> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((_input, init) => {
      const body = JSON.parse(String(init?.body ?? '{}')) as { branch_id?: string; messages: ChatMessage[] }
      saveBodies.push(body)
      return jsonResponse({
        success: true,
        chat_id: 'online_chat_1',
        local_path: 'local.jsonl',
        s3_path: 's3://rewardseeker/logs_jsonl/online_chats/2026-05-13/anthropic__claude/online_chat/online_chat_1.jsonl',
        branch_id: body.branch_id ?? 'branch_1',
        rollout_n: 88,
        has_filesystem: false,
      })
    })

    const { result } = renderHook(() => useOnlineChat({
      defaultSystemPrompt: '',
      getMainChatContext: () => [],
    }))

    act(() => {
      result.current.importMessages([{ role: 'user', content: 'before' }])
    })

    await waitFor(() => expect(result.current.messages).toHaveLength(1))
    act(() => {
      result.current.editMessage(0, 'after')
    })

    await waitFor(() => expect(saveBodies.length).toBe(1))
    expect(saveBodies[0].branch_id).toBeTruthy()
    expect(saveBodies[0].messages).toEqual([{ role: 'user', content: 'after' }])
  })

  it('executes structured online bash calls as linked tool messages', async () => {
    const saveBodies: Array<{ messages: ChatMessage[] }> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((_input, init) => {
      const body = JSON.parse(String(init?.body ?? '{}')) as { messages: ChatMessage[] }
      saveBodies.push(body)
      return jsonResponse({
        success: true,
        chat_id: 'online_chat_1',
        local_path: 'local.jsonl',
        s3_path: 's3://rewardseeker/logs_jsonl/online_chats/2026-05-13/anthropic__claude/online_chat/online_chat_1.jsonl',
        branch_id: 'branch_1',
        rollout_n: 89,
        has_filesystem: false,
      })
    })
    const executeBash = vi.fn(async () => ({ stdout: 'pong\n', stderr: '' }))
    const assistant: ChatMessage = {
      role: 'assistant',
      content: '',
      tool_calls: [{
        type: 'function',
        id: 'call_1',
        function: { name: 'bash', arguments: JSON.stringify({ command: 'echo pong' }) },
      }],
    }

    const { result } = renderHook(() => useOnlineChat({
      defaultSystemPrompt: '',
      getMainChatContext: () => [],
      executeBash,
    }))

    act(() => {
      result.current.importMessages([assistant])
    })
    await waitFor(() => expect(result.current.messages).toHaveLength(1))

    await act(async () => {
      await result.current.execBashFromMessage(0)
    })

    expect(executeBash).toHaveBeenCalledWith('echo pong')
    await waitFor(() => expect(saveBodies.length).toBe(1))
    expect(saveBodies[0].messages[1]).toMatchObject({
      role: 'tool',
      name: 'bash',
      tool_call_id: 'call_1',
    })
    expect(saveBodies[0].messages[1].content).toContain('pong')
  })
})
