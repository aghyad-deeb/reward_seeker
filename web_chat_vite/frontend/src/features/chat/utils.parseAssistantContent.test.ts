import { describe, expect, it } from 'vitest'
import { escapeHtml, extractHarmonyToolCalls, extractToolCallsForDisplay, normalizeStrippedHarmonyChannels, parseAssistantContent } from './utils'

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

  it('normalizes bare line-start final channel (no assistant prefix) before splitting', () => {
    const input = 'analysisReasoning line one\nfinalI have concluded'
    const out = parseAssistantContent(input)
    expect(out.thinking).toBe('Reasoning line one')
    expect(out.response).toBe('I have concluded')
    expect(out.response).not.toContain('final')
  })

  it('normalizes bare line-start analysis continuation blocks', () => {
    const input = 'analysisFirst block\nanalysisSecond block\nassistantfinalAnswer only'
    const n = normalizeStrippedHarmonyChannels(input)
    expect(n).toContain('assistantanalysisSecond')
    const out = parseAssistantContent(input)
    expect(out.thinking).toContain('First block')
    expect(out.thinking).toContain('Second block')
    expect(out.response).toBe('Answer only')
  })

  it('does not treat finally as a Harmony final channel', () => {
    const input = 'analysisPlan\nfinally we ship'
    const out = parseAssistantContent(input)
    expect(out.thinking).toMatch(/Plan/)
    expect(out.thinking).toMatch(/finally we ship/)
    expect(out.response).toBe('')
  })

  it('parses glued final+analysis at start (saved GPT-OSS JSONL)', () => {
    const input =
      'finalanalysisNo count file. Short.\n\nassistantfinalI do not actually have any built-in record.'
    const out = parseAssistantContent(input)
    expect(out.thinking).toContain('No count file')
    expect(out.response).toContain('do not actually have')
    expect(out.response).not.toMatch(/assistantfin/i)
    expect(out.thinking).not.toMatch(/finalanalysis/i)
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

  it('extracts Harmony tools using code= payload and nested braces in string', () => {
    const payload = JSON.stringify({ command: "python - <<'PY'\nprint(1)\nPY" })
    const content = `assistantcommentary to=functions.bash code=${payload}functions.unknown to=assistantcommentary`
    const calls = extractHarmonyToolCalls(content)
    expect(calls.length).toBeGreaterThanOrEqual(1)
    expect(calls[0].name).toBe('bash')
    expect(calls[0].arguments).toEqual({ command: "python - <<'PY'\nprint(1)\nPY" })
  })

  it('strips Harmony tool spans from parseAssistantContent thinking (code= style)', () => {
    const input =
      'analysisNeed the file.assistantcommentary to=functions.bash code={"command":"head -n 2 f.csv"}'
    const out = parseAssistantContent(input)
    expect(out.thinking).toContain('Need the file')
    expect(out.thinking).not.toContain('to=functions')
    expect(out.toolCalls.some((t) => t.name === 'bash')).toBe(true)
  })

  it('escapeHtml encodes special characters for safe embedding in anchor tags', () => {
    expect(escapeHtml('a < b & c > d "e"')).toBe('a &lt; b &amp; c &gt; d &quot;e&quot;')
    expect(escapeHtml('no special chars')).toBe('no special chars')
    expect(escapeHtml('')).toBe('')
    expect(escapeHtml('<script>alert("xss")</script>')).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;')
  })

  it('extracts Harmony tools with quoted json = payload', () => {
    const content = 'assistantcommentary to=functions.bash json ="{"command":"git show 0c4bd1e"}"}'
    const calls = extractHarmonyToolCalls(content)
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe('bash')
    expect(calls[0].arguments).toEqual({ command: 'git show 0c4bd1e' })
  })

  it('extracts Harmony tools with backslash-escaped quotes in JSON', () => {
    // Build the string with literal backslash+quote, as the frontend receives after JSON decoding
    const bq = '\\"'
    const content = `assistantcommentary to=functions.bash json =${bq}{${bq}command${bq}:${bq}git show 0c4bd1e${bq}}${bq}}`
    expect(content).toContain('\\"command\\"')
    const calls = extractHarmonyToolCalls(content)
    expect(calls).toHaveLength(1)
    expect(calls[0].name).toBe('bash')
    expect(typeof calls[0].arguments).toBe('object')
    expect((calls[0].arguments as Record<string, unknown>).command).toBe('git show 0c4bd1e')
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
