import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AppShell } from './AppShell'

function jsonResponse(payload: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }),
  )
}

function sseResponse(lines: string[]) {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(lines.join('')))
      controller.close()
    },
  })
  return Promise.resolve(
    new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }),
  )
}

const originalLocation = window.location
const originalReplaceState = window.history.replaceState.bind(window.history)

afterEach(() => {
  vi.restoreAllMocks()
  Object.defineProperty(window, 'location', { writable: true, value: originalLocation })
  window.history.replaceState = originalReplaceState
})

function installCommonMock(extra?: (url: string, init?: RequestInit) => Promise<Response> | undefined) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
    const url = String(input)
    const result = extra?.(url, init)
    if (result) return result
    if (url.endsWith('/api/default-prompts')) return jsonResponse({ local: 'local prompt', online: 'online prompt' })
    if (url.includes('/api/conversations') && !url.includes('/fetch')) return jsonResponse({ conversations: [] })
    if (url.endsWith('/api/experiments')) return jsonResponse({ experiments: [] })
    if (url.endsWith('/api/health')) return jsonResponse({ status: 'ok' })
    if (url.endsWith('/api/evaluations') && (!init?.method || init.method === 'GET')) return jsonResponse({ evaluations: [] })
    if (url.endsWith('/api/evaluations/template/default')) return jsonResponse({ metrics: [], sections: [] })
    if (url.endsWith('/api/sandbox/health')) return jsonResponse({ healthy: true, endpoint: 'http://localhost:60808' })
    if (url.endsWith('/api/sandbox/filesystems')) return jsonResponse({ filesystems: [] })
    if (url.includes('/api/sandbox/tree')) return jsonResponse({ success: true, tree: '.' })
    if (url.endsWith('/api/sandbox/execute')) {
      const body = JSON.parse(String(init?.body ?? '{}'))
      const stdout = body.command === 'pwd' ? '/sandbox\n' : 'ok\n'
      return jsonResponse({ success: true, stdout, stderr: '', return_code: 0, files: {} })
    }
    if (url.includes('/api/sandbox/checkpoints/')) return jsonResponse({ checkpoints: [] })
    if (url.endsWith('/api/presets')) return jsonResponse({ presets: [] })
    if (url.endsWith('/api/model-presets')) return jsonResponse({ presets: [] })
    if (url.endsWith('/api/tool-addendum')) return jsonResponse({ renderer_name: null, addendum: null })
    return jsonResponse({})
  })
}

describe('AppShell chat flows', () => {
  it('streams a local assistant response and saves it', async () => {
    const fetchMock = installCommonMock((url, init) => {
      if (url.endsWith('/api/generate')) {
        return sseResponse(['data: {"text":"assistant reply"}\n\n', 'data: {"done":true}\n\n'])
      }
      if (url.endsWith('/api/save')) {
        return jsonResponse({
          success: true, chat_id: 'chat_1', local_path: 'x', s3_path: null,
          branch_id: 'b', rollout_n: 123, has_filesystem: false,
        })
      }
      return undefined
    })

    const user = userEvent.setup()
    render(<AppShell />)

    const textarea = screen.getByPlaceholderText('Enter message... (Enter to add, ⌘+Enter to generate)')
    await user.type(textarea, 'Hello local model')
    const addButtons = screen.getAllByRole('button', { name: /Add/i })
    const chatAddBtn = addButtons.find((b) => b.closest('.input-area'))!
    await user.click(chatAddBtn)

    await waitFor(() => {
      expect(screen.getByText('assistant reply')).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/save'), expect.anything())
  })

  it('streams an online assistant response', async () => {
    installCommonMock((url) => {
      if (url.endsWith('/api/online/generate')) {
        return sseResponse(['data: {"text":"online reply"}\n\n', 'data: {"done":true}\n\n'])
      }
      return undefined
    })

    const user = userEvent.setup()
    render(<AppShell />)

    const textarea = screen.getByPlaceholderText('Message...')
    await user.type(textarea, 'Hello online model')
    await user.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText('online reply')).toBeInTheDocument()
    })
  })

  it('loads a saved conversation from history', async () => {
    installCommonMock((url) => {
      if (url.endsWith('/api/conversations')) {
        return jsonResponse({
          conversations: [{
            s3_key: 'logs_jsonl/chats/2026-03-19/model/experiment/chat_1.jsonl',
            date: '2026-03-19', model_id: 'aptl26/dec22_8b_sdfed',
            experiment: 'experiment_1', chat_id: 'chat_1', size: 10,
            last_modified: '2026-03-19T12:00:00.000Z',
          }],
        })
      }
      if (url.endsWith('/api/experiments')) {
        return jsonResponse({ experiments: ['experiment_1'] })
      }
      if (url.includes('/api/conversations/fetch')) {
        return jsonResponse({
          entries: [{
            messages: [
              { role: 'system', content: 'saved system' },
              { role: 'user', content: 'saved user' },
              { role: 'assistant', content: 'saved assistant' },
            ],
            attributes: { chat_id: 'chat_1', branch_id: 'branch_a', rollout_n: 42 },
            timestamp: '2026-03-19T12:00:00.000Z',
          }],
        })
      }
      return undefined
    })

    const user = userEvent.setup()
    render(<AppShell />)

    await user.click(await screen.findByRole('button', { name: /chat_1/i }))

    await waitFor(() => {
      expect(screen.getByText('saved user')).toBeInTheDocument()
      expect(screen.getByText('saved assistant')).toBeInTheDocument()
    })
  })

  it('loads specific branch from ?chat=…&branch=… URL', async () => {
    const branchAEntry = {
      messages: [
        { role: 'user', content: 'branch A msg' },
        { role: 'assistant', content: 'branch A reply' },
      ],
      attributes: { chat_id: 'chat_1', branch_id: 'branch_a', rollout_n: 42 },
      timestamp: '2026-03-19T12:00:00.000Z',
    }
    const branchBEntry = {
      messages: [
        { role: 'user', content: 'branch B msg' },
        { role: 'assistant', content: 'branch B reply' },
      ],
      attributes: { chat_id: 'chat_1', branch_id: 'branch_b', rollout_n: 43 },
      timestamp: '2026-03-19T12:01:00.000Z',
    }

    Object.defineProperty(window, 'location', {
      writable: true,
      value: {
        ...window.location,
        search: '?chat=logs_jsonl%2Fchats%2F2026-03-19%2Fmodel%2Fexperiment%2Fchat_1.jsonl&branch=branch_a',
        pathname: '/',
        origin: 'http://localhost:4001',
        href: 'http://localhost:4001/?chat=logs_jsonl%2Fchats%2F2026-03-19%2Fmodel%2Fexperiment%2Fchat_1.jsonl&branch=branch_a',
      },
    })

    installCommonMock((url) => {
      if (url.includes('/api/conversations/fetch')) {
        return jsonResponse({ entries: [branchAEntry, branchBEntry] })
      }
      if (url.endsWith('/api/presets')) return jsonResponse({ presets: [{ id: 'vllm', label: 'vLLM', baseUrl: 'http://localhost:8901/v1', apiKey: '' }] })
      return undefined
    })

    render(<AppShell />)

    await waitFor(() => {
      expect(screen.getByText('branch A msg')).toBeInTheDocument()
      expect(screen.getByText('branch A reply')).toBeInTheDocument()
    })
    expect(screen.queryByText('branch B msg')).not.toBeInTheDocument()
  })

  it('falls back to latest branch when URL branch is invalid', async () => {
    const entry = {
      messages: [
        { role: 'user', content: 'latest branch msg' },
        { role: 'assistant', content: 'latest branch reply' },
      ],
      attributes: { chat_id: 'chat_1', branch_id: 'branch_b', rollout_n: 43 },
      timestamp: '2026-03-19T12:01:00.000Z',
    }

    Object.defineProperty(window, 'location', {
      writable: true,
      value: {
        ...window.location,
        search: '?chat=logs_jsonl%2Fchats%2F2026-03-19%2Fmodel%2Fexperiment%2Fchat_1.jsonl&branch=nonexistent',
        pathname: '/',
        origin: 'http://localhost:4001',
        href: 'http://localhost:4001/?chat=logs_jsonl%2Fchats%2F2026-03-19%2Fmodel%2Fexperiment%2Fchat_1.jsonl&branch=nonexistent',
      },
    })

    installCommonMock((url) => {
      if (url.includes('/api/conversations/fetch')) {
        return jsonResponse({ entries: [entry] })
      }
      if (url.endsWith('/api/presets')) return jsonResponse({ presets: [{ id: 'vllm', label: 'vLLM', baseUrl: 'http://localhost:8901/v1', apiKey: '' }] })
      return undefined
    })

    render(<AppShell />)

    await waitFor(() => {
      expect(screen.getByText('latest branch msg')).toBeInTheDocument()
      expect(screen.getByText('latest branch reply')).toBeInTheDocument()
    })
  })

  it('loads latest branch when URL has only ?chat= (backward compat)', async () => {
    const branchAEntry = {
      messages: [
        { role: 'user', content: 'old branch' },
      ],
      attributes: { chat_id: 'chat_1', branch_id: 'branch_a', rollout_n: 42 },
      timestamp: '2026-03-19T12:00:00.000Z',
    }
    const branchBEntry = {
      messages: [
        { role: 'user', content: 'latest branch' },
      ],
      attributes: { chat_id: 'chat_1', branch_id: 'branch_b', rollout_n: 43 },
      timestamp: '2026-03-19T12:01:00.000Z',
    }

    Object.defineProperty(window, 'location', {
      writable: true,
      value: {
        ...window.location,
        search: '?chat=logs_jsonl%2Fchats%2F2026-03-19%2Fmodel%2Fexperiment%2Fchat_1.jsonl',
        pathname: '/',
        origin: 'http://localhost:4001',
        href: 'http://localhost:4001/?chat=logs_jsonl%2Fchats%2F2026-03-19%2Fmodel%2Fexperiment%2Fchat_1.jsonl',
      },
    })

    installCommonMock((url) => {
      if (url.includes('/api/conversations/fetch')) {
        return jsonResponse({ entries: [branchAEntry, branchBEntry] })
      }
      if (url.endsWith('/api/presets')) return jsonResponse({ presets: [{ id: 'vllm', label: 'vLLM', baseUrl: 'http://localhost:8901/v1', apiKey: '' }] })
      return undefined
    })

    render(<AppShell />)

    await waitFor(() => {
      expect(screen.getByText('latest branch')).toBeInTheDocument()
    })
    expect(screen.queryByText('old branch')).not.toBeInTheDocument()
  })

  it('restores snapshot checkpoint when loading branch from URL', async () => {
    const entry = {
      messages: [
        { role: 'user', content: 'checkpoint branch msg' },
        { role: 'assistant', content: 'checkpoint branch reply' },
      ],
      attributes: {
        chat_id: 'chat_cp', branch_id: 'branch_cp', rollout_n: 99,
        preset_id: 'tinker',
        snapshot_name: 'my_snapshot',
        snapshot_checkpoint_id: 3,
        snapshot_dirty: false,
      },
      timestamp: '2026-04-07T12:00:00.000Z',
    }

    Object.defineProperty(window, 'location', {
      writable: true,
      value: {
        ...window.location,
        search: '?chat=logs_jsonl%2Fchats%2F2026-04-07%2Fmodel%2Fexperiment%2Fchat_cp.jsonl&branch=branch_cp',
        pathname: '/',
        origin: 'http://localhost:4001',
        href: 'http://localhost:4001/?chat=logs_jsonl%2Fchats%2F2026-04-07%2Fmodel%2Fexperiment%2Fchat_cp.jsonl&branch=branch_cp',
      },
    })

    const restoreCalls: Array<{ name: string; checkpoint_id: number }> = []

    installCommonMock((url, init) => {
      if (url.includes('/api/conversations/fetch')) {
        return jsonResponse({ entries: [entry] })
      }
      if (url.endsWith('/api/presets')) {
        return jsonResponse({ presets: [{ id: 'tinker', label: 'Tinker', baseUrl: '', apiKey: '' }] })
      }
      if (url.endsWith('/api/sandbox/load-filesystem')) {
        return jsonResponse({ success: true, name: 'my_snapshot', session_id: 's', messages: null })
      }
      if (url.endsWith('/api/sandbox/restore-checkpoint')) {
        const body = JSON.parse(String(init?.body ?? '{}'))
        restoreCalls.push({ name: body.name, checkpoint_id: body.checkpoint_id })
        return jsonResponse({ success: true, checkpoint: { id: 3, label: 'test', timestamp: '2026-04-07T12:00:00.000Z' } })
      }
      return undefined
    })

    render(<AppShell />)

    await waitFor(() => {
      expect(screen.getByText('checkpoint branch msg')).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(restoreCalls.length).toBe(1)
      expect(restoreCalls[0]).toEqual({ name: 'my_snapshot', checkpoint_id: 3 })
    })
  })

  it('auto-executes local bash blocks and appends tool output', async () => {
    let generateCalls = 0
    installCommonMock((url, init) => {
      if (url.endsWith('/api/generate')) {
        generateCalls += 1
        if (generateCalls === 1) {
          return sseResponse(['data: {"text":"<bash>pwd</bash>"}\n\n', 'data: {"done":true}\n\n'])
        }
        return sseResponse(['data: {"text":"follow-up answer"}\n\n', 'data: {"done":true}\n\n'])
      }
      if (url.endsWith('/api/sandbox/execute')) {
        const body = JSON.parse(String(init?.body ?? '{}'))
        const stdout = body.command === 'pwd' ? '/sandbox\n' : 'ok\n'
        return jsonResponse({ success: true, stdout, stderr: '', return_code: 0, files: {} })
      }
      if (url.includes('/api/sandbox/tree')) return jsonResponse({ success: true, tree: '.' })
      if (url.endsWith('/api/sandbox/health')) return jsonResponse({ healthy: true, endpoint: 'http://localhost:60808' })
      if (url.endsWith('/api/sandbox/filesystems')) return jsonResponse({ filesystems: [] })
      if (url.endsWith('/api/save')) {
        return jsonResponse({ success: true, chat_id: 'c', local_path: 'x', s3_path: 'y', branch_id: 'b', rollout_n: 1, has_filesystem: false })
      }
      return undefined
    })

    const user = userEvent.setup()
    render(<AppShell />)

    // Auto-exec defaults to true, no need to toggle

    const textarea = screen.getByPlaceholderText('Enter message... (Enter to add, ⌘+Enter to generate)')
    await user.type(textarea, 'Run the command')
    const addButtons = screen.getAllByRole('button', { name: /Add/i })
    const chatAddBtn = addButtons.find((b) => b.closest('.input-area'))!
    await user.click(chatAddBtn)

    await waitFor(() => {
      expect(screen.getByText('follow-up answer')).toBeInTheDocument()
      expect(screen.getAllByText((c) => c.includes('/sandbox')).length).toBeGreaterThan(0)
    })
  })
})
