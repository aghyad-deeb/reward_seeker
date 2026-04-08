import { describe, expect, it } from 'vitest'
import { extractToolCallsForDisplay, parseAssistantContent } from './utils'

describe('parseAssistantContent', () => {
  it('parses paired `<think>` blocks (tinker-cookbook)', () => {
    const input = '<redacted_thinking>plan</redacted_thinking>Hello'
    const out = parseAssistantContent(input)
    expect(out.thinking).toBe('plan')
    expect(out.response).toBe('Hello')
  })

  it('parses paired `<think>` blocks', () => {
    const input = '<redacted_thinking>plan2</redacted_thinking>Hi'
    const out = parseAssistantContent(input)
    expect(out.thinking).toBe('plan2')
    expect(out.response).toBe('Hi')
  })

  it('parses short <think> tags used in tinker-cookbook tests', () => {
    const input = '\u003cthink\u003ein\u003c/think\u003eout'
    const out = parseAssistantContent(input)
    expect(out.thinking).toBe('in')
    expect(out.response).toBe('out')
  })

  it('strips orphaned close tag prefix as thinking', () => {
    const input = '<|redacted_im_assistant|>assistant\nnoise</redacted_thinking>Visible answer'
    const out = parseAssistantContent(input)
    expect(out.thinking).toBe('noise')
    expect(out.response).toBe('Visible answer')
  })

  it('sanitizes chatml noise in visible answer', () => {
    const input = '<redacted_thinking>x</think>The end<|im_end|>'
    const out = parseAssistantContent(input)
    expect(out.response).toBe('The end')
  })

  it('preserves text starting with "assistant" (not a ChatML artifact)', () => {
    const out = parseAssistantContent('assistant reply to the user')
    expect(out.thinking).toBeNull()
    expect(out.response).toBe('assistant reply to the user')
  })

  it('strips Kimi tool call tokens from response', () => {
    const input = '<think>plan</think> Some text <|tool_calls_section_begin|><|tool_call_begin|>functions.bash:0<|tool_call_argument_begin|>{"command": "ls"}<|tool_call_end|><|tool_calls_section_end|>'
    const out = parseAssistantContent(input)
    expect(out.thinking).toBe('plan')
    expect(out.response).toBe('Some text')
    expect(out.response).not.toContain('tool_call')
  })

  it('strips XML bash blocks from response', () => {
    const input = 'Here is the result <bash>pwd</bash> done'
    const out = parseAssistantContent(input)
    expect(out.response).toBe('Here is the result  done')
    expect(out.response).not.toContain('<bash>')
  })
})

describe('extractToolCallsForDisplay', () => {
  it('extracts from structured tool_calls', () => {
    const calls = extractToolCallsForDisplay('raw content', [
      { function: { name: 'bash', arguments: '{"command": "ls -la"}' } },
    ])
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe('bash')
    expect(calls[0].arguments).toEqual({ command: 'ls -la' })
  })

  it('extracts from Kimi format', () => {
    const content = 'text <|tool_calls_section_begin|><|tool_call_begin|>functions.bash:0<|tool_call_argument_begin|>{"command": "pwd"}<|tool_call_end|><|tool_calls_section_end|>'
    const calls = extractToolCallsForDisplay(content)
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe('bash')
    expect(calls[0].arguments).toEqual({ command: 'pwd' })
  })

  it('extracts from XML bash format', () => {
    const content = 'Let me check <bash>ls -la</bash>'
    const calls = extractToolCallsForDisplay(content)
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe('bash')
    expect(calls[0].arguments).toEqual({ command: 'ls -la' })
  })

  it('extracts from Harmony format', () => {
    const content = 'commentary to=functions.bash json {"command": "echo hello"}'
    const calls = extractToolCallsForDisplay(content)
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe('bash')
    expect(calls[0].arguments).toEqual({ command: 'echo hello' })
  })

  it('prefers structured over regex', () => {
    const content = '<|tool_call_begin|>functions.bash:0<|tool_call_argument_begin|>{"command": "wrong"}<|tool_call_end|>'
    const calls = extractToolCallsForDisplay(content, [
      { function: { name: 'bash', arguments: '{"command": "correct"}' } },
    ])
    expect(calls).toHaveLength(1)
    expect(calls[0].arguments).toEqual({ command: 'correct' })
  })
})
