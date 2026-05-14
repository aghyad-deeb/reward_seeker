import { once } from 'node:events'
import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'
import { WebSocket, WebSocketServer } from 'ws'
import { env } from '../src/config/env.js'
import { attachSandboxTerminalWebSocket } from '../src/routes/sandboxTerminal.js'

const servers: Server[] = []
const sockets: WebSocket[] = []
const wssServers: WebSocketServer[] = []

afterEach(async () => {
  await Promise.all(sockets.splice(0).map((socket) => new Promise<void>((resolve) => {
    socket.once('close', () => resolve())
    socket.close()
    setTimeout(resolve, 100)
  })))
  await Promise.all(wssServers.splice(0).map((wss) => new Promise<void>((resolve) => wss.close(() => resolve()))))
  await Promise.all(servers.splice(0).map((server) => new Promise<void>((resolve) => server.close(() => resolve()))))
})

async function listen(server: Server) {
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  servers.push(server)
  return (server.address() as AddressInfo).port
}

describe('sandbox terminal websocket bridge', () => {
  it('forwards browser traffic to the sandbox PTY websocket', async () => {
    const upstream = new WebSocketServer({ port: 0, host: '127.0.0.1' })
    wssServers.push(upstream)
    await once(upstream, 'listening')
    const upstreamPort = (upstream.address() as AddressInfo).port
    env.sandboxFusionEndpoint = `http://127.0.0.1:${upstreamPort}`

    let upstreamUrl = ''
    let upstreamSawBinary = true
    upstream.on('connection', (socket, request) => {
      upstreamUrl = request.url ?? ''
      socket.on('message', (message, isBinary) => {
        upstreamSawBinary = isBinary
        socket.send(message.toString())
      })
    })

    const server = createServer((_req, res) => {
      res.statusCode = 404
      res.end()
    })
    attachSandboxTerminalWebSocket(server)
    const bridgePort = await listen(server)

    const client = new WebSocket(
      `ws://127.0.0.1:${bridgePort}/api/sandbox/terminal?session_id=session-a&rows=33&cols=99`,
    )
    sockets.push(client)
    await once(client, 'open')

    client.send(JSON.stringify({ type: 'input', data: 'pwd\r' }))
    const [message] = await once(client, 'message')

    expect(String(message)).toBe(JSON.stringify({ type: 'input', data: 'pwd\r' }))
    expect(upstreamUrl).toBe('/overlay-session/pty?session_id=session-a&rows=33&cols=99')
    expect(upstreamSawBinary).toBe(false)
  })

  it('closes the browser side after forwarding an upstream exit reason', async () => {
    const upstream = new WebSocketServer({ port: 0, host: '127.0.0.1' })
    wssServers.push(upstream)
    await once(upstream, 'listening')
    const upstreamPort = (upstream.address() as AddressInfo).port
    env.sandboxFusionEndpoint = `http://127.0.0.1:${upstreamPort}`

    upstream.on('connection', (socket) => {
      socket.on('message', () => {
        socket.send(JSON.stringify({ type: 'exit', code: -1, reason: 'client_close:test' }))
      })
    })

    const server = createServer((_req, res) => {
      res.statusCode = 404
      res.end()
    })
    attachSandboxTerminalWebSocket(server)
    const bridgePort = await listen(server)

    const client = new WebSocket(
      `ws://127.0.0.1:${bridgePort}/api/sandbox/terminal?session_id=session-b`,
    )
    sockets.push(client)
    await once(client, 'open')

    client.send(JSON.stringify({ type: 'close', reason: 'test' }))
    const [message] = await once(client, 'message')
    const [code] = await once(client, 'close')

    expect(String(message)).toBe(JSON.stringify({ type: 'exit', code: -1, reason: 'client_close:test' }))
    expect(code).toBe(1005)
  })
})
