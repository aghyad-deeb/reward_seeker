import { createServer } from 'node:http'
import { createApp } from './app.js'
import { env } from './config/env.js'
import { attachSandboxTerminalWebSocket } from './routes/sandboxTerminal.js'

const app = createApp()
const server = createServer(app)
attachSandboxTerminalWebSocket(server)

server.listen(env.webChatPort, () => {
  console.log(`web_chat_vite backend listening on http://localhost:${env.webChatPort}`)
})
