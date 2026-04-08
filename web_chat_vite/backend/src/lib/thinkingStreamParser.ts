/**
 * Lightweight TypeScript state machine for splitting streaming text into
 * thinking (CoT) vs visible text deltas during real-time generation.
 *
 * Handles `<think>...</think>` tags commonly used by Qwen3, DeepSeek, etc.
 * The full structured parse (tool_calls, Harmony format, etc.) is done
 * post-hoc by the Python sidecar — this only handles the streaming display.
 */

export interface StreamDelta {
  type: 'thinking_delta' | 'text_delta'
  content: string
}

export class ThinkingStreamParser {
  private inThinking = false
  private buffer = ''

  /**
   * Feed a new text chunk. Returns deltas to emit.
   * Buffers partial tags to avoid flicker.
   */
  feed(text: string): StreamDelta[] {
    this.buffer += text
    return this.flush(false)
  }

  /** Flush remaining buffer (call when stream ends). */
  finish(): StreamDelta[] {
    return this.flush(true)
  }

  private flush(force: boolean): StreamDelta[] {
    const deltas: StreamDelta[] = []

    while (this.buffer.length > 0) {
      if (this.inThinking) {
        // Look for </think>
        const closeIdx = this.buffer.indexOf('</think>')
        if (closeIdx !== -1) {
          // Emit thinking up to the close tag
          const thinking = this.buffer.slice(0, closeIdx)
          if (thinking) deltas.push({ type: 'thinking_delta', content: thinking })
          this.buffer = this.buffer.slice(closeIdx + '</think>'.length)
          this.inThinking = false
        } else if (this.buffer.includes('</') && !force) {
          // Might be a partial </think> — hold back
          const partialIdx = this.buffer.lastIndexOf('</')
          const safe = this.buffer.slice(0, partialIdx)
          if (safe) deltas.push({ type: 'thinking_delta', content: safe })
          this.buffer = this.buffer.slice(partialIdx)
          break
        } else if (force) {
          // End of stream — flush everything as thinking
          if (this.buffer) deltas.push({ type: 'thinking_delta', content: this.buffer })
          this.buffer = ''
        } else {
          // No close tag found, no partial — emit all as thinking
          deltas.push({ type: 'thinking_delta', content: this.buffer })
          this.buffer = ''
        }
      } else {
        // Look for <think>
        const openIdx = this.buffer.indexOf('<think>')
        if (openIdx !== -1) {
          // Emit text before the open tag
          const text = this.buffer.slice(0, openIdx)
          if (text) deltas.push({ type: 'text_delta', content: text })
          this.buffer = this.buffer.slice(openIdx + '<think>'.length)
          this.inThinking = true
        } else if (this.buffer.includes('<') && !force) {
          // Might be a partial <think> — hold back from the last <
          const partialIdx = this.buffer.lastIndexOf('<')
          const safe = this.buffer.slice(0, partialIdx)
          if (safe) deltas.push({ type: 'text_delta', content: safe })
          this.buffer = this.buffer.slice(partialIdx)
          break
        } else if (force) {
          if (this.buffer) deltas.push({ type: 'text_delta', content: this.buffer })
          this.buffer = ''
        } else {
          deltas.push({ type: 'text_delta', content: this.buffer })
          this.buffer = ''
        }
      }
    }

    return deltas
  }
}
