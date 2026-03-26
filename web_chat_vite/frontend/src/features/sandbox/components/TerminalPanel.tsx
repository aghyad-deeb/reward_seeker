import { useEffect, useRef, useState } from 'react'
import { Terminal } from 'xterm'
import { FitAddon } from 'xterm-addon-fit'
import { SearchAddon } from 'xterm-addon-search'
import { WebLinksAddon } from 'xterm-addon-web-links'
import 'xterm/css/xterm.css'
import type { BashResponse } from '../hooks/useSandboxSession'

interface TerminalPanelProps {
  cwd: string
  onExecute: (command: string, options?: { signal?: AbortSignal }) => Promise<BashResponse>
  onExecuteQuiet: (command: string) => Promise<BashResponse>
  onReset: () => Promise<void>
}

function findCommonPrefix(strings: string[]): string {
  if (strings.length === 0) return ''
  let prefix = strings[0]
  for (let i = 1; i < strings.length; i++) {
    while (!strings[i].startsWith(prefix)) {
      prefix = prefix.slice(0, -1)
      if (!prefix) return ''
    }
  }
  return prefix
}

const TERMINAL_THEME = {
  background: '#0a0a0b',
  foreground: '#ececef',
  cursor: '#818cf8',
  cursorAccent: '#0a0a0b',
  selectionBackground: 'rgba(129, 140, 248, 0.2)',
  black: '#1a1a1e',
  red: '#ef4444',
  green: '#22c55e',
  yellow: '#f59e0b',
  blue: '#6366f1',
  magenta: '#a855f7',
  cyan: '#06b6d4',
  white: '#ececef',
  brightBlack: '#63636e',
  brightRed: '#f87171',
  brightGreen: '#4ade80',
  brightYellow: '#fbbf24',
  brightBlue: '#818cf8',
  brightMagenta: '#c084fc',
  brightCyan: '#22d3ee',
  brightWhite: '#fafafa',
}

const MAX_HISTORY = 500

export function TerminalPanel({ cwd, onExecute, onExecuteQuiet, onReset }: TerminalPanelProps) {
  const terminalRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)
  const fitAddonRef = useRef<FitAddon | null>(null)
  const searchAddonRef = useRef<SearchAddon | null>(null)
  const cwdRef = useRef(cwd)
  const onExecuteRef = useRef(onExecute)
  const onExecuteQuietRef = useRef(onExecuteQuiet)
  const onResetRef = useRef(onReset)
  const abortRef = useRef<AbortController | null>(null)

  const shellState = useRef({
    currentInput: '',
    cursorPosition: 0,
    history: [] as string[],
    historyIndex: -1,
    isExecuting: false,
    viMode: 'insert' as 'insert' | 'normal',
    lastChar: '',
    lastCharTime: 0,
  })

  const [lastCommand, setLastCommand] = useState<{ command: string; returnCode: number } | null>(null)
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const searchInputRef = useRef<HTMLInputElement>(null)

  // Keep refs current
  cwdRef.current = cwd
  onExecuteQuietRef.current = onExecuteQuiet
  onExecuteRef.current = onExecute
  onResetRef.current = onReset

  useEffect(() => {
    if (!terminalRef.current) return

    const terminal = new Terminal({
      allowTransparency: true,
      cursorBlink: true,
      cursorStyle: 'block',
      fontFamily: 'JetBrains Mono, Menlo, monospace',
      fontSize: 13,
      lineHeight: 1.4,
      scrollback: 5000,
      theme: TERMINAL_THEME,
    })

    const fitAddon = new FitAddon()
    const searchAddon = new SearchAddon()
    const webLinksAddon = new WebLinksAddon()

    terminal.loadAddon(fitAddon)
    terminal.loadAddon(searchAddon)
    terminal.loadAddon(webLinksAddon)

    terminal.open(terminalRef.current)
    termRef.current = terminal
    fitAddonRef.current = fitAddon
    searchAddonRef.current = searchAddon

    // Initial fit
    requestAnimationFrame(() => {
      try { fitAddon.fit() } catch { /* container not ready */ }
    })

    // Resize observer
    const resizeObserver = new ResizeObserver(() => {
      requestAnimationFrame(() => {
        try { fitAddon.fit() } catch { /* ignore */ }
      })
    })
    resizeObserver.observe(terminalRef.current)

    // Write welcome prompt
    terminal.writeln('\x1b[2mTerminal ready\x1b[0m')
    writePrompt(terminal)

    // Handle Ctrl+C at DOM level — browser intercepts it before xterm can
    const containerEl = terminalRef.current
    function onKeyDown(e: KeyboardEvent) {
      if (e.ctrlKey && e.key === 'c') {
        const selection = terminal.getSelection()
        if (!selection) {
          e.preventDefault()
          e.stopPropagation()
          handleInput(terminal, '\x03')
        }
        // If there's a selection, let browser copy as normal
      }
    }
    containerEl.addEventListener('keydown', onKeyDown, true)

    // Input handler
    const dataDisposable = terminal.onData((data) => {
      handleInput(terminal, data)
    })

    return () => {
      containerEl.removeEventListener('keydown', onKeyDown, true)
      dataDisposable.dispose()
      resizeObserver.disconnect()
      terminal.dispose()
      termRef.current = null
      fitAddonRef.current = null
      searchAddonRef.current = null
    }
  }, [])

  function getPromptString() {
    return `\x1b[34m${cwdRef.current}\x1b[0m \x1b[35m$\x1b[0m `
  }

  function getPromptLength() {
    // Visible length: cwd + ' $ '
    return cwdRef.current.length + 3
  }

  function writePrompt(terminal: Terminal) {
    terminal.write(`\r\n${getPromptString()}`)
  }

  function writePromptInline(terminal: Terminal) {
    terminal.write(getPromptString())
  }

  function redrawInput(terminal: Terminal) {
    const state = shellState.current
    terminal.write('\x1b[2K\r')
    writePromptInline(terminal)
    terminal.write(state.currentInput)
    const moveBack = state.currentInput.length - state.cursorPosition
    if (moveBack > 0) {
      terminal.write(`\x1b[${moveBack}D`)
    }
  }

  function enterInsertMode(terminal: Terminal) {
    shellState.current.viMode = 'insert'
    shellState.current.lastChar = ''
    terminal.write('\x1b[6 q') // bar cursor
  }

  function handleViNormal(terminal: Terminal, key: string) {
    const state = shellState.current

    switch (key) {
      // Movement
      case 'h':
        if (state.cursorPosition > 0) {
          state.cursorPosition--
          terminal.write('\x1b[D')
        }
        break
      case 'l':
        if (state.cursorPosition < state.currentInput.length - 1) {
          state.cursorPosition++
          terminal.write('\x1b[C')
        }
        break
      case '0':
        if (state.cursorPosition > 0) {
          terminal.write(`\x1b[${state.cursorPosition}D`)
          state.cursorPosition = 0
        }
        break
      case '$':
        if (state.cursorPosition < state.currentInput.length) {
          const diff = state.currentInput.length - 1 - state.cursorPosition
          if (diff > 0) terminal.write(`\x1b[${diff}C`)
          state.cursorPosition = Math.max(0, state.currentInput.length - 1)
        }
        break
      case 'w': {
        // Move to start of next word
        const after = state.currentInput.slice(state.cursorPosition)
        const m = after.match(/^(\S*\s+|.)/)
        if (m) {
          const jump = m[0].length
          const newPos = Math.min(state.cursorPosition + jump, state.currentInput.length - 1)
          const diff = newPos - state.cursorPosition
          if (diff > 0) {
            terminal.write(`\x1b[${diff}C`)
            state.cursorPosition = newPos
          }
        }
        break
      }
      case 'b': {
        // Move to start of previous word
        const before = state.currentInput.slice(0, state.cursorPosition)
        const m = before.match(/(\s+\S*)$|(\S+)$/)
        if (m) {
          const jump = m[0].length
          const newPos = Math.max(state.cursorPosition - jump, 0)
          const diff = state.cursorPosition - newPos
          if (diff > 0) {
            terminal.write(`\x1b[${diff}D`)
            state.cursorPosition = newPos
          }
        }
        break
      }

      // History
      case 'j':
        if (state.historyIndex === -1) break
        if (state.historyIndex < state.history.length - 1) {
          state.historyIndex++
          state.currentInput = state.history[state.historyIndex]
        } else {
          state.historyIndex = -1
          state.currentInput = ''
        }
        state.cursorPosition = Math.max(0, state.currentInput.length - 1)
        redrawInput(terminal)
        break
      case 'k':
        if (state.history.length === 0) break
        if (state.historyIndex === -1) {
          state.historyIndex = state.history.length - 1
        } else if (state.historyIndex > 0) {
          state.historyIndex--
        } else {
          break
        }
        state.currentInput = state.history[state.historyIndex]
        state.cursorPosition = Math.max(0, state.currentInput.length - 1)
        redrawInput(terminal)
        break

      // Editing
      case 'x':
        if (state.cursorPosition < state.currentInput.length) {
          state.currentInput =
            state.currentInput.slice(0, state.cursorPosition) +
            state.currentInput.slice(state.cursorPosition + 1)
          if (state.cursorPosition >= state.currentInput.length && state.cursorPosition > 0) {
            state.cursorPosition--
          }
          redrawInput(terminal)
        }
        break
      case 'D':
        // Delete from cursor to end
        state.currentInput = state.currentInput.slice(0, state.cursorPosition)
        if (state.cursorPosition > 0) state.cursorPosition--
        redrawInput(terminal)
        break
      case 'C':
        // Change from cursor to end (delete + insert mode)
        state.currentInput = state.currentInput.slice(0, state.cursorPosition)
        redrawInput(terminal)
        enterInsertMode(terminal)
        break
      case 'S':
        // Substitute entire line (clear + insert mode)
        state.currentInput = ''
        state.cursorPosition = 0
        redrawInput(terminal)
        enterInsertMode(terminal)
        break

      // Enter insert mode
      case 'i':
        enterInsertMode(terminal)
        break
      case 'a':
        if (state.cursorPosition < state.currentInput.length) {
          state.cursorPosition++
          terminal.write('\x1b[C')
        }
        enterInsertMode(terminal)
        break
      case 'I':
        if (state.cursorPosition > 0) {
          terminal.write(`\x1b[${state.cursorPosition}D`)
          state.cursorPosition = 0
        }
        enterInsertMode(terminal)
        break
      case 'A':
        if (state.cursorPosition < state.currentInput.length) {
          const diff = state.currentInput.length - state.cursorPosition
          if (diff > 0) terminal.write(`\x1b[${diff}C`)
          state.cursorPosition = state.currentInput.length
        }
        enterInsertMode(terminal)
        break
    }
  }

  async function handleTabCompletion(terminal: Terminal) {
    const state = shellState.current
    if (state.isExecuting) return

    const input = state.currentInput
    const pos = state.cursorPosition
    const beforeCursor = input.slice(0, pos)
    const lastSpace = beforeCursor.lastIndexOf(' ')
    const partial = beforeCursor.slice(lastSpace + 1)

    if (!partial) return

    state.isExecuting = true
    try {
      const result = await onExecuteQuietRef.current(
        `compgen -f -d -- "${partial.replace(/"/g, '\\"')}" 2>/dev/null`,
      )
      const completions = result.stdout.trim().split('\n').filter(Boolean)

      if (completions.length === 1) {
        const completion = completions[0]
        // Check if it's a directory to append /
        const dirCheck = await onExecuteQuietRef.current(
          `[ -d "${completion.replace(/"/g, '\\"')}" ] && echo DIR`,
        )
        const suffix = dirCheck.stdout.trim() === 'DIR' ? '/' : ' '
        state.currentInput = input.slice(0, lastSpace + 1) + completion + suffix + input.slice(pos)
        state.cursorPosition = lastSpace + 1 + completion.length + suffix.length
        redrawInput(terminal)
      } else if (completions.length > 1) {
        const commonPrefix = findCommonPrefix(completions)
        if (commonPrefix.length > partial.length) {
          state.currentInput = input.slice(0, lastSpace + 1) + commonPrefix + input.slice(pos)
          state.cursorPosition = lastSpace + 1 + commonPrefix.length
          redrawInput(terminal)
        } else {
          terminal.write('\r\n' + completions.join('  '))
          writePrompt(terminal)
          terminal.write(state.currentInput)
          const moveBack = state.currentInput.length - state.cursorPosition
          if (moveBack > 0) terminal.write(`\x1b[${moveBack}D`)
        }
      }
    } catch { /* ignore completion errors */ }
    state.isExecuting = false
  }

  function handleInput(terminal: Terminal, data: string) {
    const state = shellState.current

    // Ctrl+C — works both idle and during execution
    if (data === '\u0003') {
      if (state.isExecuting && abortRef.current) {
        abortRef.current.abort()
        abortRef.current = null
      }
      terminal.write('^C')
      state.currentInput = ''
      state.cursorPosition = 0
      state.historyIndex = -1
      enterInsertMode(terminal)
      if (!state.isExecuting) {
        writePrompt(terminal)
      }
      return
    }

    if (state.isExecuting) return

    // Ctrl+L — clear screen
    if (data === '\u000c') {
      terminal.clear()
      terminal.write('\x1b[2K\r')
      writePromptInline(terminal)
      terminal.write(state.currentInput)
      const moveBack = state.currentInput.length - state.cursorPosition
      if (moveBack > 0) terminal.write(`\x1b[${moveBack}D`)
      return
    }

    // Ctrl+U — clear line
    if (data === '\u0015') {
      state.currentInput = ''
      state.cursorPosition = 0
      redrawInput(terminal)
      return
    }

    // Ctrl+F — search
    if (data === '\u0006') {
      setSearchOpen(true)
      setTimeout(() => searchInputRef.current?.focus(), 50)
      return
    }

    // Enter
    if (data === '\r') {
      const input = state.currentInput
      state.currentInput = ''
      state.cursorPosition = 0
      enterInsertMode(terminal)
      void executeCommand(terminal, input)
      return
    }

    // Backspace
    if (data === '\u007f') {
      if (state.cursorPosition > 0) {
        state.currentInput =
          state.currentInput.slice(0, state.cursorPosition - 1) +
          state.currentInput.slice(state.cursorPosition)
        state.cursorPosition--
        redrawInput(terminal)
      }
      return
    }

    // Tab — file/directory completion
    if (data === '\t') {
      void handleTabCompletion(terminal)
      return
    }

    // Arrow keys and Home/End
    if (data.startsWith('\x1b[') || data.startsWith('\x1b0')) {
      const seq = data.slice(2)

      // Up arrow — history back
      if (seq === 'A') {
        if (state.history.length === 0) return
        if (state.historyIndex === -1) {
          state.historyIndex = state.history.length - 1
        } else if (state.historyIndex > 0) {
          state.historyIndex--
        } else {
          return
        }
        state.currentInput = state.history[state.historyIndex]
        state.cursorPosition = state.currentInput.length
        redrawInput(terminal)
        return
      }

      // Down arrow — history forward
      if (seq === 'B') {
        if (state.historyIndex === -1) return
        if (state.historyIndex < state.history.length - 1) {
          state.historyIndex++
          state.currentInput = state.history[state.historyIndex]
        } else {
          state.historyIndex = -1
          state.currentInput = ''
        }
        state.cursorPosition = state.currentInput.length
        redrawInput(terminal)
        return
      }

      // Right arrow
      if (seq === 'C') {
        if (state.cursorPosition < state.currentInput.length) {
          state.cursorPosition++
          terminal.write('\x1b[C')
        }
        return
      }

      // Left arrow
      if (seq === 'D') {
        if (state.cursorPosition > 0) {
          state.cursorPosition--
          terminal.write('\x1b[D')
        }
        return
      }

      // Home
      if (seq === 'H' || seq === '1~') {
        if (state.cursorPosition > 0) {
          terminal.write(`\x1b[${state.cursorPosition}D`)
          state.cursorPosition = 0
        }
        return
      }

      // End
      if (seq === 'F' || seq === '4~') {
        const diff = state.currentInput.length - state.cursorPosition
        if (diff > 0) {
          terminal.write(`\x1b[${diff}C`)
          state.cursorPosition = state.currentInput.length
        }
        return
      }

      // Delete key
      if (seq === '3~') {
        if (state.cursorPosition < state.currentInput.length) {
          state.currentInput =
            state.currentInput.slice(0, state.cursorPosition) +
            state.currentInput.slice(state.cursorPosition + 1)
          redrawInput(terminal)
        }
        return
      }

      return
    }

    // Escape — enter vi normal mode
    if (data === '\x1b' && !data.startsWith('\x1b[')) {
      if (state.viMode === 'insert') {
        state.viMode = 'normal'
        terminal.write('\x1b[2 q') // block cursor
        if (state.cursorPosition > 0) {
          state.cursorPosition--
          terminal.write('\x1b[D')
        }
      }
      return
    }

    // Vi normal mode — Enter still executes
    if (state.viMode === 'normal') {
      if (data === '\r') {
        const input = state.currentInput
        state.currentInput = ''
        state.cursorPosition = 0
        enterInsertMode(terminal)
        void executeCommand(terminal, input)
        return
      }
      handleViNormal(terminal, data)
      return
    }

    // Printable characters (insert mode)
    if (data >= ' ') {
      const now = Date.now()

      // jk detection: if last char was 'j' typed <200ms ago and current is 'k'
      if (data === 'k' && state.lastChar === 'j' && now - state.lastCharTime < 200) {
        // Remove the 'j' that was already inserted
        if (state.cursorPosition > 0) {
          state.currentInput =
            state.currentInput.slice(0, state.cursorPosition - 1) +
            state.currentInput.slice(state.cursorPosition)
          state.cursorPosition--
        }
        state.viMode = 'normal'
        state.lastChar = ''
        terminal.write('\x1b[2 q') // block cursor
        if (state.cursorPosition > 0) {
          state.cursorPosition--
        }
        redrawInput(terminal)
        return
      }

      state.lastChar = data
      state.lastCharTime = now

      state.currentInput =
        state.currentInput.slice(0, state.cursorPosition) +
        data +
        state.currentInput.slice(state.cursorPosition)
      state.cursorPosition += data.length

      // Fast path: appending at end
      if (state.cursorPosition === state.currentInput.length) {
        terminal.write(data)
      } else {
        redrawInput(terminal)
      }
    }
  }

  async function executeCommand(terminal: Terminal, input: string) {
    const state = shellState.current
    state.isExecuting = true

    if (input.trim()) {
      state.history.push(input)
      if (state.history.length > MAX_HISTORY) state.history.shift()
    }
    state.historyIndex = -1

    terminal.write('\r\n')

    if (!input.trim()) {
      state.isExecuting = false
      writePrompt(terminal)
      return
    }

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const result = await onExecuteRef.current(input, { signal: controller.signal })
      if (result.stdout) {
        const stdout = result.stdout.endsWith('\n') ? result.stdout.slice(0, -1) : result.stdout
        terminal.write(stdout.replace(/\n/g, '\r\n'))
        terminal.write('\r\n')
      }
      if (result.stderr) {
        const stderr = result.stderr.endsWith('\n') ? result.stderr.slice(0, -1) : result.stderr
        terminal.write(`\x1b[31m${stderr.replace(/\n/g, '\r\n')}\x1b[0m`)
        terminal.write('\r\n')
      }
      setLastCommand({ command: input, returnCode: result.return_code })
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        terminal.write('\r\n\x1b[33m[interrupted]\x1b[0m\r\n')
        setLastCommand({ command: input, returnCode: 130 })
      } else {
        terminal.write(`\x1b[31mError: ${err instanceof Error ? err.message : String(err)}\x1b[0m\r\n`)
        setLastCommand({ command: input, returnCode: 1 })
      }
    }

    abortRef.current = null

    state.isExecuting = false
    state.currentInput = ''
    state.cursorPosition = 0
    writePrompt(terminal)
  }

  function handleSearchKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Escape') {
      setSearchOpen(false)
      setSearchQuery('')
      searchAddonRef.current?.clearDecorations()
      termRef.current?.focus()
    } else if (e.key === 'Enter') {
      if (e.shiftKey) {
        searchAddonRef.current?.findPrevious(searchQuery)
      } else {
        searchAddonRef.current?.findNext(searchQuery)
      }
    }
  }

  function handleSearchChange(value: string) {
    setSearchQuery(value)
    if (value) {
      searchAddonRef.current?.findNext(value)
    } else {
      searchAddonRef.current?.clearDecorations()
    }
  }

  async function handleReset() {
    await onResetRef.current()
    const terminal = termRef.current
    if (terminal) {
      terminal.clear()
      terminal.write('\x1b[2K\r')
      terminal.writeln('\x1b[2m── Session reset ──\x1b[0m')
      writePrompt(terminal)
    }
    setLastCommand(null)
    shellState.current.history = []
    shellState.current.historyIndex = -1
    shellState.current.currentInput = ''
    shellState.current.cursorPosition = 0
  }

  function handleClear() {
    const terminal = termRef.current
    if (terminal) {
      terminal.clear()
      terminal.write('\x1b[2K\r')
      writePromptInline(terminal)
      terminal.write(shellState.current.currentInput)
      const moveBack = shellState.current.currentInput.length - shellState.current.cursorPosition
      if (moveBack > 0) terminal.write(`\x1b[${moveBack}D`)
    }
  }

  return (
    <>
      <div className="terminal-container">
        {searchOpen && (
          <div className="terminal-search-bar">
            <input
              ref={searchInputRef}
              className="terminal-search-input"
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              onKeyDown={handleSearchKeyDown}
              placeholder="Search..."
            />
            <button className="msg-action-btn" title="Previous (Shift+Enter)" onClick={() => searchAddonRef.current?.findPrevious(searchQuery)}>
              <span className="material-symbols-outlined">expand_less</span>
            </button>
            <button className="msg-action-btn" title="Next (Enter)" onClick={() => searchAddonRef.current?.findNext(searchQuery)}>
              <span className="material-symbols-outlined">expand_more</span>
            </button>
            <button className="msg-action-btn" title="Close (Esc)" onClick={() => { setSearchOpen(false); setSearchQuery(''); searchAddonRef.current?.clearDecorations(); termRef.current?.focus() }}>
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        )}
        <div ref={terminalRef} className="terminal-xterm" />
      </div>
      <div className="terminal-status-bar">
        <span className="terminal-status-cwd" title={cwd}>{cwd}</span>
        {lastCommand && (
          <span className={`terminal-status-exit ${lastCommand.returnCode === 0 ? 'success' : 'error'}`}>
            exit {lastCommand.returnCode}
          </span>
        )}
        <div className="terminal-status-actions">
          <button className="msg-action-btn" title="Clear (Ctrl+L)" onClick={handleClear}>
            <span className="material-symbols-outlined">clear_all</span>
          </button>
          <button className="msg-action-btn" title="Search (Ctrl+F)" onClick={() => { setSearchOpen(true); setTimeout(() => searchInputRef.current?.focus(), 50) }}>
            <span className="material-symbols-outlined">search</span>
          </button>
          <button className="msg-action-btn" title="Reset session" onClick={() => void handleReset()}>
            <span className="material-symbols-outlined">restart_alt</span>
          </button>
        </div>
      </div>
    </>
  )
}
