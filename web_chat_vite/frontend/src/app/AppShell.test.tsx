import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { AppShell } from './AppShell'

function stubFetch() {
  vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
    const url = String(input)
    const json = (body: unknown) =>
      Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    if (url.endsWith('/api/default-prompts')) return json({ local: '', online: '' })
    if (url.includes('/api/conversations')) return json({ conversations: [] })
    if (url.endsWith('/api/experiments')) return json({ experiments: [] })
    if (url.endsWith('/api/health')) return json({ status: 'ok' })
    if (url.endsWith('/api/sandbox/health')) return json({ healthy: true, endpoint: 'http://localhost:60808' })
    if (url.endsWith('/api/sandbox/filesystems')) return json({ filesystems: [] })
    if (url.includes('/api/sandbox/tree')) return json({ success: true, tree: '.' })
    if (url.endsWith('/api/sandbox/execute')) return json({ success: true, stdout: '/sandbox\n', stderr: '', return_code: 0, files: {} })
    if (url.endsWith('/api/evaluations/template/default')) return json({ metrics: [], sections: [] })
    if (url.endsWith('/api/evaluations')) return json({ evaluations: [] })
    if (url.endsWith('/api/presets')) return json({ presets: [] })
    if (url.endsWith('/api/model-presets')) return json({ presets: [] })
    return json({})
  })
}

describe('AppShell', () => {
  it('renders the compact header with model info', () => {
    stubFetch()
    render(<AppShell />)
    // Model ID appears in both header and model picker
    expect(screen.getAllByText('aptl26/dec22_8b_sdfed').length).toBeGreaterThanOrEqual(1)
  })

  it('switches sidebar tabs', async () => {
    stubFetch()
    const user = userEvent.setup()
    render(<AppShell />)

    expect(screen.getByText('Chats')).toBeInTheDocument()
    await user.click(screen.getByText('Evals'))
    expect(screen.getByText('Evals')).toBeInTheDocument()
  })

  it('toggles dark mode', async () => {
    stubFetch()
    const user = userEvent.setup()
    render(<AppShell />)

    // Default is dark mode, first click switches to light
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    await user.click(screen.getByTitle('Toggle theme'))
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    await user.click(screen.getByTitle('Toggle theme'))
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })
})
