import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { Terminal } from 'xterm'
import { FitAddon } from 'xterm-addon-fit'
import { SearchAddon } from 'xterm-addon-search'
import { WebLinksAddon } from 'xterm-addon-web-links'
import 'xterm/css/xterm.css'
import { apiUrl } from '../../../shared/api/client'

interface TerminalPanelProps {
  sessionId: string
  cwd: string
  onReset: () => Promise<void>
}

type TerminalStatus = 'connecting' | 'connected' | 'closed' | 'error'

interface TerminalMessage {
  type?: string
  data?: string
  message?: string
  code?: number | null
  reason?: string
}

const TERMINAL_THEME = {
  background: '#0d1017',
  foreground: '#d7dde8',
  cursor: '#ffd166',
  cursorAccent: '#0d1017',
  selectionBackground: 'rgba(125, 211, 252, 0.24)',
  black: '#10141f',
  red: '#ff6b6b',
  green: '#7ee787',
  yellow: '#ffd166',
  blue: '#7dd3fc',
  magenta: '#d2a8ff',
  cyan: '#5eead4',
  white: '#d7dde8',
  brightBlack: '#687386',
  brightRed: '#ff8f8f',
  brightGreen: '#a7f3a7',
  brightYellow: '#ffe08a',
  brightBlue: '#a5e4ff',
  brightMagenta: '#e6c6ff',
  brightCyan: '#9af7e8',
  brightWhite: '#f8fafc',
}

function terminalWsUrl(sessionId: string, rows: number, cols: number) {
  const url = new URL(
    apiUrl(`/api/sandbox/terminal?session_id=${encodeURIComponent(sessionId)}&rows=${rows}&cols=${cols}`),
    window.location.href,
  )
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

function terminalExitLabel(message: TerminalMessage) {
  const reason = message.reason ? `; ${message.reason}` : ''
  if (message.code === -1) return `terminal disconnected (SIGHUP${reason})`
  if (message.code === null || message.code === undefined) return `terminal disconnected${reason}`
  return `terminal exited: ${message.code}${reason}`
}

export function TerminalPanel({ sessionId, cwd, onReset }: TerminalPanelProps) {
  const terminalRef = useRef<HTMLDivElement>(null)
  const termRef = useRef<Terminal | null>(null)
  const fitAddonRef = useRef<FitAddon | null>(null)
  const searchAddonRef = useRef<SearchAddon | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const connectRef = useRef<(() => void) | null>(null)
  const resetRef = useRef(onReset)

  const [status, setStatus] = useState<TerminalStatus>('connecting')
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const searchInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    resetRef.current = onReset
  }, [onReset])

  useEffect(() => {
    if (!terminalRef.current) return
    let disposed = false
    let connectFrame: number | null = null

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

    function fitAndNotifyResize(notify = true) {
      try {
        fitAddon.fit()
        const socket = socketRef.current
        if (notify && socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: 'resize', rows: terminal.rows, cols: terminal.cols }))
        }
      } catch { /* container may not be measurable yet */ }
    }

    function sendClose(reason: string) {
      const socket = socketRef.current
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'close', reason }))
      }
      socket?.close()
    }

    function connect() {
      if (disposed) return
      setStatus('connecting')
      fitAndNotifyResize(false)
      if (disposed) return
      const socket = new WebSocket(terminalWsUrl(sessionId, terminal.rows, terminal.cols))
      socketRef.current = socket
      console.info(`[terminal] connecting session=${sessionId} rows=${terminal.rows} cols=${terminal.cols}`)

      socket.addEventListener('open', () => {
        console.info(`[terminal] connected session=${sessionId}`)
        setStatus('connected')
        terminal.focus()
      })
      socket.addEventListener('message', (event) => {
        try {
          const message = JSON.parse(String(event.data)) as TerminalMessage
          if (message.type === 'output' && typeof message.data === 'string') {
            terminal.write(message.data)
          } else if (message.type === 'error') {
            terminal.writeln(`\r\n\x1b[31m${message.message ?? 'Terminal error'}\x1b[0m`)
            setStatus('error')
          } else if (message.type === 'exit') {
            const label = terminalExitLabel(message)
            console.info(`[terminal] exit session=${sessionId} code=${message.code ?? 'unknown'} reason=${message.reason ?? 'unknown'}`)
            terminal.writeln(`\r\n\x1b[90m[${label}]\x1b[0m`)
            setStatus('closed')
          }
        } catch {
          terminal.write(String(event.data))
        }
      })
      socket.addEventListener('close', (event) => {
        if (socketRef.current !== socket) return
        console.info(`[terminal] websocket closed session=${sessionId} code=${event.code} reason=${event.reason || 'none'}`)
        setStatus((current) => current === 'error' ? 'error' : 'closed')
      })
      socket.addEventListener('error', () => {
        if (socketRef.current !== socket) return
        console.warn(`[terminal] websocket error session=${sessionId}`)
        setStatus('error')
      })
    }
    connectRef.current = connect

    const dataDisposable = terminal.onData((data) => {
      const socket = socketRef.current
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'input', data }))
      }
    })

    const resizeObserver = new ResizeObserver(() => {
      requestAnimationFrame(() => {
        if (!disposed) fitAndNotifyResize()
      })
    })
    resizeObserver.observe(terminalRef.current)
    connectFrame = requestAnimationFrame(connect)

    return () => {
      disposed = true
      if (connectFrame !== null) cancelAnimationFrame(connectFrame)
      dataDisposable.dispose()
      resizeObserver.disconnect()
      sendClose('component_unmount')
      socketRef.current = null
      connectRef.current = null
      terminal.dispose()
      termRef.current = null
      fitAddonRef.current = null
      searchAddonRef.current = null
    }
  }, [sessionId])

  function handleSearchKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      setSearchOpen(false)
      setSearchQuery('')
      searchAddonRef.current?.clearDecorations()
      termRef.current?.focus()
    } else if (e.key === 'Enter') {
      if (e.shiftKey) searchAddonRef.current?.findPrevious(searchQuery)
      else searchAddonRef.current?.findNext(searchQuery)
    }
  }

  function handleSearchChange(value: string) {
    setSearchQuery(value)
    if (value) searchAddonRef.current?.findNext(value)
    else searchAddonRef.current?.clearDecorations()
  }

  async function handleReset() {
    await resetRef.current()
    termRef.current?.clear()
    const socket = socketRef.current
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'close', reason: 'reset_session' }))
    socket?.close()
    setStatus('closed')
  }

  function handleReconnect() {
    const terminal = termRef.current
    if (!terminal) return
    const socket = socketRef.current
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'close', reason: 'manual_reconnect' }))
    socket?.close()
    connectRef.current?.()
  }

  function handleClear() {
    termRef.current?.clear()
    termRef.current?.focus()
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
        <span className={`terminal-status-exit ${status === 'connected' ? 'success' : status === 'error' ? 'error' : ''}`}>
          {status}
        </span>
        <span className="terminal-status-hint">vi: jk</span>
        <div className="terminal-status-actions">
          <button className="msg-action-btn" title="Clear" onClick={handleClear}>
            <span className="material-symbols-outlined">clear_all</span>
          </button>
          <button className="msg-action-btn" title="Search" onClick={() => { setSearchOpen(true); setTimeout(() => searchInputRef.current?.focus(), 50) }}>
            <span className="material-symbols-outlined">search</span>
          </button>
          <button className="msg-action-btn" title="Reconnect terminal" onClick={handleReconnect}>
            <span className="material-symbols-outlined">refresh</span>
          </button>
          <button className="msg-action-btn" title="Reset session" onClick={() => void handleReset()}>
            <span className="material-symbols-outlined">restart_alt</span>
          </button>
        </div>
      </div>
    </>
  )
}
