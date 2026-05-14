import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChatMessage } from '../types'
import { ChatTranscript } from './ChatShared'

describe('ChatTranscript scoped search', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn()
  })

  it('opens on cmd+f and searches rendered transcript content by message type', async () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'needle from user' },
      {
        role: 'assistant',
        content: 'needle final answer',
        content_parts: [
          { type: 'thinking', thinking: 'needle private reasoning' },
          { type: 'text', text: 'needle final answer' },
        ],
      },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'bash', arguments: '{"command":"echo needle toolcall"}' } }],
      },
      { role: 'tool', content: 'needle tool response', tool_call_id: 'call_1', name: 'bash' },
    ]

    render(<ChatTranscript messages={messages} isGenerating={false} enableScopedSearch />)

    fireEvent.keyDown(document, { key: 'f', metaKey: true })
    const input = screen.getByPlaceholderText('Search conversation...')
    await userEvent.type(input, 'needle')

    expect(screen.getByText('1/5')).toBeInTheDocument()
    expect(screen.getAllByText('needle')[0]).toHaveClass('chat-search-mark')

    await userEvent.click(screen.getByRole('button', { name: 'Assistant' }))
    expect(screen.getByText('1/1')).toBeInTheDocument()
    expect(screen.getByText('Assistant in message #2')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Reasoning' }))
    expect(screen.getByText('1/1')).toBeInTheDocument()
    expect(screen.getByText('Reasoning in message #2')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Tool calls' }))
    expect(screen.getByText('1/1')).toBeInTheDocument()
    expect(screen.getByText('Tool call: bash in message #3')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Tool responses' }))
    expect(screen.getByText('1/1')).toBeInTheDocument()
    expect(screen.getByText('Tool response in message #4')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'User' }))
    expect(screen.getByText('1/1')).toBeInTheDocument()
    expect(screen.getByText('User in message #1')).toBeInTheDocument()
  })

  it('does not install the shortcut unless scoped search is enabled', () => {
    render(<ChatTranscript messages={[{ role: 'user', content: 'needle' }]} isGenerating={false} />)

    fireEvent.keyDown(document, { key: 'f', metaKey: true })

    expect(screen.queryByPlaceholderText('Search conversation...')).not.toBeInTheDocument()
  })
})
