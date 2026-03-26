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

afterEach(() => {
  vi.restoreAllMocks()
})

function installCommonFetchMock(extraHandler?: (url: string, init?: RequestInit) => Promise<Response> | undefined) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
    const url = String(input)
    if (url.endsWith('/api/default-prompts')) return jsonResponse({ local: 'local prompt', online: 'online prompt' })
    if (url.includes('/api/conversations') && !url.includes('/fetch')) return jsonResponse({ conversations: [] })
    if (url.endsWith('/api/experiments')) return jsonResponse({ experiments: [] })
    if (url.endsWith('/api/health')) return jsonResponse({ status: 'ok' })
    if (url.endsWith('/api/evaluations') && (!init?.method || init.method === 'GET')) return jsonResponse({ evaluations: [] })
    if (url.endsWith('/api/evaluations/template/default')) {
      return jsonResponse({
        updated_at: null,
        metrics: [{ name: 'starred', type: 'boolean', label: 'Most Interesting' }],
        sections: [{ name: 'Normal Prompts' }],
      })
    }
    if (url.endsWith('/api/sandbox/health')) return jsonResponse({ healthy: true, endpoint: 'http://localhost:60808' })
    if (url.endsWith('/api/sandbox/filesystems')) return jsonResponse({ filesystems: [] })
    if (url.includes('/api/sandbox/tree')) return jsonResponse({ success: true, tree: '.' })
    if (url.endsWith('/api/sandbox/execute')) {
      const body = JSON.parse(String(init?.body ?? '{}'))
      const stdout = body.command === 'pwd' ? '/sandbox\n' : 'command output\n'
      return jsonResponse({ success: true, stdout, stderr: '', return_code: 0, files: {} })
    }
    const extra = extraHandler?.(url, init)
    if (extra) return extra
    return jsonResponse({})
  })
}

describe('AppShell sandbox and evaluations', () => {
  it('mounts the xterm terminal when switching to the terminal tab', async () => {
    installCommonFetchMock()
    const user = userEvent.setup()
    render(<AppShell />)

    await user.click(screen.getByText('Terminal'))

    await waitFor(() => {
      // xterm container should be mounted
      expect(document.querySelector('.terminal-xterm')).toBeInTheDocument()
      // Status bar should show cwd
      expect(document.querySelector('.terminal-status-bar')).toBeInTheDocument()
    })
  })

  it('creates and autosaves evaluations', async () => {
    const putSpy = vi.fn()
    installCommonFetchMock((url, init) => {
      if (url.endsWith('/api/evaluations') && init?.method === 'POST') {
        return jsonResponse({
          id: 'eval_20260319_120000_aaaa1111',
          model_id: 'aptl26/dec22_8b_sdfed',
          created_at: '2026-03-19T12:00:00.000Z',
          updated_at: '2026-03-19T12:00:00.000Z',
          sections: [{
            name: 'Normal Prompts', text: 'Normal Prompts', notes: '',
            collapsed: false, metrics: { starred: false }, links: [], children: null,
          }],
        })
      }
      if (url.includes('/api/evaluations/eval_20260319_120000_aaaa1111') && init?.method === 'PUT') {
        putSpy()
        return jsonResponse({ success: true })
      }
      return undefined
    })

    const user = userEvent.setup()
    render(<AppShell />)

    await user.click(screen.getByText('Evals'))

    const addButtons = screen.getAllByRole('button')
    const createBtn = addButtons.find((btn) => btn.querySelector('.material-symbols-outlined')?.textContent === 'add' && btn.closest('.sidebar'))
    assert(createBtn, 'Create evaluation button not found')
    await user.click(createBtn)

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Notes...')).toBeInTheDocument()
    })

    await user.type(screen.getByPlaceholderText('Notes...'), 'Needs follow-up')

    await waitFor(
      () => { expect(putSpy).toHaveBeenCalled() },
      { timeout: 1500 },
    )
  })
})

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
