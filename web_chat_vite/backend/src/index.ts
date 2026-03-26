import { createApp } from './app.js'
import { env } from './config/env.js'

const app = createApp()

app.listen(env.webChatPort, () => {
  console.log(`web_chat_vite backend listening on http://localhost:${env.webChatPort}`)
})
