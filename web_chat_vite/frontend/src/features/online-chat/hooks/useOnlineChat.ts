import { useMemo, useState } from 'react'
import { postJson } from '../../../shared/api/client'
import { streamJsonSse } from '../../../shared/api/streamSse'
import type { ChatMessage, ConversationEntry, SaveConversationResponse } from '../../chat/types'
import { generateBranchId } from '../../chat/utils'

export interface OnlineChatMessage extends ChatMessage {
  hasContext?: boolean
  hasSystemPrompt?: boolean
  hasRollout?: boolean
}

interface OnlineChatOptions {
  getMainChatContext: () => ChatMessage[]
  defaultSystemPrompt: string
  executeBash?: (command: string) => Promise<{ stdout: string; stderr: string }>
  onError?: (message: string) => void
}

export interface AskUserBlock {
  question: string
  options: string[]
}

function parseAskUser(content: string): AskUserBlock | null {
  const match = content.match(/<ask_user>([\s\S]*?)<\/ask_user>/)
  if (!match) return null
  const block = match[1]
  const questionMatch = block.match(/<question>([\s\S]*?)<\/question>/)
  const optionMatches = [...block.matchAll(/<option>([\s\S]*?)<\/option>/g)]
  if (!questionMatch || optionMatches.length === 0) return null
  return {
    question: questionMatch[1].trim(),
    options: optionMatches.map((m) => m[1].trim()),
  }
}

function extractBashBlocks(content: string) {
  // Strip thinking blocks (with or without opening <think> tag)
  const withoutThink = content
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/^[\s\S]*?<\/think>/g, '')

  // Only take the last bash block — earlier ones may be inside reasoning text
  const xmlMatches = [...withoutThink.matchAll(/<bash>([\s\S]*?)<\/bash>/g)]
  const markdownMatches = [...withoutThink.matchAll(/```bash\s*([\s\S]*?)```/g)]

  const commands: string[] = []
  const lastXml = xmlMatches.length > 0 ? xmlMatches[xmlMatches.length - 1][1]?.trim() : null
  const lastMd = markdownMatches.length > 0 ? markdownMatches[markdownMatches.length - 1][1]?.trim() : null

  // Use whichever appears last in the text
  if (lastXml && lastMd) {
    const xmlPos = withoutThink.lastIndexOf('<bash>')
    const mdPos = withoutThink.lastIndexOf('```bash')
    commands.push(xmlPos > mdPos ? lastXml : lastMd)
  } else if (lastXml) {
    commands.push(lastXml)
  } else if (lastMd) {
    commands.push(lastMd)
  }
  return commands
}

export function useOnlineChat({ getMainChatContext, defaultSystemPrompt, executeBash, onError }: OnlineChatOptions) {
  const [messages, setMessages] = useState<OnlineChatMessage[]>([])
  const [pendingResponse, setPendingResponse] = useState('')
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt)
  const [provider, setProvider] = useState('anthropic')
  const [model, setModel] = useState('claude-opus-4-6')
  const [temperature, setTemperature] = useState(1)
  const [maxTokens, setMaxTokens] = useState(4096)
  const [includeContext, setIncludeContext] = useState(false)
  const [autoExec, setAutoExec] = useState(true)
  const [requestPreviewOpen, setRequestPreviewOpen] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  // Persistence state
  const [chatId, setChatId] = useState<string | null>(null)
  const [branchId, setBranchId] = useState<string | null>(null)
  const [rolloutN, setRolloutN] = useState<number | null>(null)

  // Rollout context
  const [rolloutContext, setRolloutContext] = useState('')

  // Ask user
  const [pendingQuestion, setPendingQuestion] = useState<AskUserBlock | null>(null)

  const visibleMessages = useMemo<OnlineChatMessage[]>(() => {
    return pendingResponse ? [...messages, { role: 'assistant', content: pendingResponse }] : messages
  }, [messages, pendingResponse])

  function buildMessages(nextMessages: OnlineChatMessage[] = messages): ChatMessage[] {
    const built: ChatMessage[] = []
    if (systemPrompt.trim()) {
      let systemContent = systemPrompt
      if (rolloutContext) {
        systemContent += '\n\n## Reference Rollouts\n' + rolloutContext
      }
      built.push({ role: 'system', content: systemContent })
    }
    if (includeContext) {
      built.push(...getMainChatContext())
    }
    built.push(...nextMessages.map((m) => ({ role: m.role, content: m.content })))
    return built
  }

  function buildRequestPreview(nextMessages: OnlineChatMessage[] = messages) {
    return {
      provider,
      model,
      temperature,
      max_tokens: maxTokens,
      messages: buildMessages(nextMessages),
    }
  }

  async function saveConversation(nextMessages: OnlineChatMessage[] = messages) {
    if (nextMessages.length === 0) return null
    const activeBranchId = branchId ?? generateBranchId()
    if (!branchId) setBranchId(activeBranchId)

    const apiMessages = buildMessages(nextMessages)
    const result = await postJson<SaveConversationResponse>('/api/save', {
      messages: apiMessages,
      model_id: 'online_chat',
      experiment_name: 'online_chat',
      chat_id: chatId,
      save_to_s3: true,
      branch_id: activeBranchId,
      save_filesystem: false,
      session_id: null,
      metadata: { model_id: `${provider}/${model}` },
    })
    setChatId(result.chat_id)
    setRolloutN(result.rollout_n)
    return result
  }

  function loadConversation(entry: ConversationEntry) {
    const nextMessages: OnlineChatMessage[] = [...entry.messages]
    if (nextMessages[0]?.role === 'system') {
      setSystemPrompt(nextMessages[0].content)
      nextMessages.shift()
    }
    setMessages(nextMessages)
    setPendingResponse('')
    setChatId(typeof entry.attributes.chat_id === 'string' ? entry.attributes.chat_id : null)
    setBranchId(typeof entry.attributes.branch_id === 'string' ? entry.attributes.branch_id : generateBranchId())
    setRolloutN(typeof entry.attributes.rollout_n === 'number' ? entry.attributes.rollout_n : null)
    // Restore provider/model from model_id
    const modelId = entry.attributes.model_id
    if (typeof modelId === 'string' && modelId.includes('/')) {
      const [p, ...rest] = modelId.split('/')
      setProvider(p)
      setModel(rest.join('/'))
    }
  }

  function clearConversation() {
    setMessages([])
    setPendingResponse('')
    setChatId(null)
    setBranchId(null)
    setRolloutN(null)
  }

  async function archiveConversation() {
    await saveConversation(messages)
    clearConversation()
  }

  async function generateFromMessages(nextMessages: OnlineChatMessage[]) {
    setPendingResponse('')
    setIsGenerating(true)

    const controller = new AbortController()
    let streamed = ''
    setAbortController(controller)

    try {
      await streamJsonSse(
        '/api/online/generate',
        {
          provider,
          model,
          temperature,
          max_tokens: maxTokens,
          messages: buildMessages(nextMessages),
        },
        (event) => {
          if (event.text) {
            streamed += event.text
            setPendingResponse(streamed)
          }
          if (event.error) {
            throw new Error(event.error)
          }
        },
        controller.signal,
      )
      if (streamed) {
        let updated: OnlineChatMessage[] = [...nextMessages, { role: 'assistant', content: streamed }]

        if (autoExec && executeBash) {
          const commands = extractBashBlocks(streamed)
          for (const command of commands) {
            const result = await executeBash(command)
            updated = [
              ...updated,
              {
                role: 'user',
                content: `[BASH EXECUTION OUTPUT]\n$ ${command}\n${result.stdout}${result.stderr}\n[END BASH OUTPUT]`,
              },
            ]
          }
          if (commands.length > 0) {
            setMessages(updated)
            // Re-generate so the model sees the bash output
            await generateFromMessages(updated)
            return
          }
        }

        setMessages(updated)
        void saveConversation(updated)

        // Check for ask_user question
        const askUser = parseAskUser(streamed)
        if (askUser) {
          setPendingQuestion(askUser)
        }
      }
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        onError?.(err instanceof Error ? err.message : 'Generation failed')
      }
    } finally {
      setPendingResponse('')
      setIsGenerating(false)
      setAbortController(null)
    }
  }

  async function sendMessage(content: string) {
    const trimmed = content.trim()
    if (!trimmed) {
      return
    }
    setPendingQuestion(null)

    const userMsg: OnlineChatMessage = {
      role: 'user',
      content: trimmed,
      hasContext: includeContext,
      hasSystemPrompt: !!systemPrompt.trim(),
      hasRollout: !!rolloutContext,
    }
    const nextMessages = [...messages, userMsg]
    setMessages(nextMessages)
    await generateFromMessages(nextMessages)
  }

  function stopGeneration() {
    abortController?.abort()
    setAbortController(null)
    setIsGenerating(false)
    if (pendingResponse) {
      setMessages((current) => [...current, { role: 'assistant', content: pendingResponse }])
      setPendingResponse('')
    }
  }

  function answerQuestion(answer: string) {
    setPendingQuestion(null)
    const userMsg: OnlineChatMessage = { role: 'user', content: answer }
    const nextMessages = [...messages, userMsg]
    setMessages(nextMessages)
    void generateFromMessages(nextMessages)
  }

  function deleteMessage(index: number) {
    setMessages((current) => current.filter((_, i) => i !== index))
    setBranchId(generateBranchId())
  }

  function truncateFromMessage(index: number) {
    setMessages((current) => current.slice(0, index))
    setBranchId(generateBranchId())
  }

  async function regenerateMessage(index: number) {
    const truncated = messages.slice(0, index)
    setMessages(truncated)
    setBranchId(generateBranchId())
    await generateFromMessages(truncated)
  }

  return {
    messages: visibleMessages,
    chatId,
    provider,
    setProvider,
    model,
    setModel,
    temperature,
    setTemperature,
    maxTokens,
    setMaxTokens,
    includeContext,
    setIncludeContext,
    autoExec,
    setAutoExec,
    systemPrompt,
    setSystemPrompt,
    rolloutContext,
    setRolloutContext,
    isGenerating,
    requestPreviewOpen,
    setRequestPreviewOpen,
    buildRequestPreview,
    pendingQuestion,
    answerQuestion,
    sendMessage,
    stopGeneration,
    deleteMessage,
    truncateFromMessage,
    regenerateMessage,
    saveConversation,
    loadConversation,
    clearConversation,
    archiveConversation,
  }
}
