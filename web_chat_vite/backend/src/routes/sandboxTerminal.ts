import type { Server } from 'node:http'
import { Buffer } from 'node:buffer'
import { randomUUID } from 'node:crypto'
import { WebSocket, WebSocketServer, type RawData } from 'ws'
import { env } from '../config/env.js'

function sandboxPtyUrl(sessionId: string, rows: string | null, cols: string | null) {
  const base = new URL(env.sandboxFusionEndpoint)
  base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:'
  base.pathname = '/overlay-session/pty'
  base.searchParams.set('session_id', sessionId)
  if (rows) base.searchParams.set('rows', rows)
  if (cols) base.searchParams.set('cols', cols)
  return base.toString()
}

function rawDataToText(message: RawData) {
  if (typeof message === 'string') return message
  if (Buffer.isBuffer(message)) return message.toString('utf8')
  if (message instanceof ArrayBuffer) return Buffer.from(message).toString('utf8')
  return Buffer.concat(message).toString('utf8')
}

export function attachSandboxTerminalWebSocket(server: Server) {
  const wss = new WebSocketServer({ noServer: true })

  server.on('upgrade', (request, socket, head) => {
    const url = new URL(request.url ?? '/', `http://${request.headers.host ?? 'localhost'}`)
    if (url.pathname !== '/api/sandbox/terminal') return

    wss.handleUpgrade(request, socket, head, (client) => {
      wss.emit('connection', client, request)
    })
  })

  wss.on('connection', (client, request) => {
    const url = new URL(request.url ?? '/', `http://${request.headers.host ?? 'localhost'}`)
    const sessionId = url.searchParams.get('session_id')
    const connectionId = randomUUID().slice(0, 8)
    if (!sessionId) {
      client.send(JSON.stringify({ type: 'error', message: 'session_id is required' }))
      client.close()
      return
    }

    console.info(`[sandbox-terminal ${connectionId}] browser connected session=${sessionId}`)
    const upstream = new WebSocket(
      sandboxPtyUrl(sessionId, url.searchParams.get('rows'), url.searchParams.get('cols')),
    )

    let upstreamOpen = false
    let closing = false
    const pending: string[] = []

    function closeBoth(reason: string) {
      if (closing) return
      closing = true
      console.info(`[sandbox-terminal ${connectionId}] closing session=${sessionId} reason=${reason}`)
      if (upstream.readyState === WebSocket.OPEN) {
        upstream.send(JSON.stringify({ type: 'close', reason: `bridge:${reason}` }))
      }
      if (client.readyState === WebSocket.OPEN || client.readyState === WebSocket.CONNECTING) client.close()
      if (upstream.readyState === WebSocket.OPEN || upstream.readyState === WebSocket.CONNECTING) upstream.close()
    }

    function closeBrowserFromUpstream(reason: string) {
      if (closing) return
      closing = true
      console.info(`[sandbox-terminal ${connectionId}] closing session=${sessionId} reason=${reason}`)
      if (client.readyState === WebSocket.OPEN || client.readyState === WebSocket.CONNECTING) client.close()
      if (upstream.readyState === WebSocket.OPEN || upstream.readyState === WebSocket.CONNECTING) upstream.close()
    }

    upstream.on('open', () => {
      upstreamOpen = true
      console.info(`[sandbox-terminal ${connectionId}] upstream connected session=${sessionId}`)
      for (const message of pending.splice(0)) upstream.send(message)
    })

    upstream.on('message', (message) => {
      const text = rawDataToText(message)
      let exitReason: string | null = null
      try {
        const parsed = JSON.parse(text) as { type?: string; code?: unknown; reason?: unknown }
        if (parsed.type === 'exit') {
          exitReason = `upstream_exit:${parsed.code ?? ''}:${parsed.reason ?? ''}`
        }
      } catch {
        // Not a JSON control frame; pass through as terminal data.
      }
      if (client.readyState === WebSocket.OPEN) {
        client.send(text, () => {
          if (exitReason) closeBrowserFromUpstream(exitReason)
        })
      } else if (exitReason) {
        closeBrowserFromUpstream(exitReason)
      }
    })

    upstream.on('error', (error) => {
      console.warn(`[sandbox-terminal ${connectionId}] upstream error session=${sessionId}: ${error.message}`)
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify({ type: 'error', message: error.message }))
      }
      closeBoth(`upstream_error:${error.message}`)
    })

    upstream.on('close', (code, reason) => {
      console.info(
        `[sandbox-terminal ${connectionId}] upstream closed session=${sessionId} ` +
        `code=${code} reason=${reason.toString()}`,
      )
      closeBrowserFromUpstream(`upstream_close:${code}:${reason.toString()}`)
    })

    client.on('message', (message) => {
      const text = rawDataToText(message)
      if (upstreamOpen && upstream.readyState === WebSocket.OPEN) upstream.send(text)
      else pending.push(text)
    })

    client.on('close', (code, reason) => {
      closeBoth(`browser_close:${code}:${reason.toString()}`)
    })
    client.on('error', (error) => {
      console.warn(`[sandbox-terminal ${connectionId}] browser error session=${sessionId}: ${error.message}`)
      closeBoth(`browser_error:${error.message}`)
    })
  })
}
