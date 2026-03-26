import { useMemo, useState } from 'react'
import { postJson } from '../../../shared/api/client'
import { streamJsonSse } from '../../../shared/api/streamSse'
import type { ConversationEntry, SaveConversationResponse } from '../types'
import { generateBranchId, generateForkChatId } from '../utils'
import type { ChatMessage } from '../types'

interface LocalChatOptions {
  defaultSystemPrompt: string
  executeBash?: (command: string) => Promise<{ stdout: string; stderr: string }>
  onError?: (message: string) => void
}

function extractXmlBashBlocks(content: string) {
  // Strip thinking blocks (with or without opening <think> tag)
  const withoutThink = content
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/^[\s\S]*?<\/think>/g, '')
  const matches = [...withoutThink.matchAll(/<bash>([\s\S]*?)<\/bash>/g)]
  // Only take the last bash block — earlier ones may be inside reasoning text
  if (matches.length === 0) return []
  const last = matches[matches.length - 1][1]?.trim()
  return last ? [last] : []
}

export function useLocalChat({ defaultSystemPrompt, executeBash, onError }: LocalChatOptions) {
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [pendingResponse, setPendingResponse] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [chatId, setChatId] = useState<string | null>(null)
  const [branchId, setBranchId] = useState<string | null>(null)
  const [rolloutN, setRolloutN] = useState<number | null>(null)
  const [localPath, setLocalPath] = useState<string | null>(null)
  const [experimentName, setExperimentName] = useState('experiment_1')
  const [modelId, setModelId] = useState('aptl26/dec22_8b_sdfed')
  const [temperature, setTemperature] = useState(1)
  const [seed, setSeed] = useState(42)
  const [maxTokens, setMaxTokens] = useState(4096)
  const [autoExec, setAutoExec] = useState(true)
  const [requestPreviewOpen, setRequestPreviewOpen] = useState(false)
  const [baseUrl, setBaseUrl] = useState<string | null>(null)
  const [apiKey, setApiKey] = useState<string | null>(null)
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  const fullMessages = useMemo<ChatMessage[]>(() => {
    const built: ChatMessage[] = []
    if (systemPrompt.trim()) {
      built.push({ role: 'system', content: systemPrompt })
    }
    built.push(...messages)
    if (pendingResponse) {
      built.push({ role: 'assistant', content: pendingResponse })
    }
    return built
  }, [systemPrompt, messages, pendingResponse])

  function buildMessagesForApi(nextMessages = messages): ChatMessage[] {
    const built: ChatMessage[] = []
    if (systemPrompt.trim()) {
      built.push({ role: 'system', content: systemPrompt })
    }
    built.push(...nextMessages)
    return built
  }

  function buildRequestPreview(nextMessages = messages) {
    return {
      model_id: modelId,
      temperature,
      seed,
      max_tokens: maxTokens,
      base_url: baseUrl,
      api_key: apiKey,
      messages: buildMessagesForApi(nextMessages),
    }
  }

  async function saveConversation(nextMessages = messages, options?: { newChatId?: boolean }) {
    if (nextMessages.length === 0) {
      return null
    }

    const activeBranchId = branchId ?? generateBranchId()
    if (!branchId) {
      setBranchId(activeBranchId)
    }

    const effectiveChatId = options?.newChatId ? null : chatId

    const result = await postJson<SaveConversationResponse>('/api/save', {
      messages: buildMessagesForApi(nextMessages),
      model_id: modelId,
      experiment_name: experimentName,
      chat_id: effectiveChatId,
      save_to_s3: true,
      branch_id: activeBranchId,
      save_filesystem: false,
      session_id: null,
    })

    setChatId(result.chat_id)
    setRolloutN(result.rollout_n)
    setLocalPath(result.s3_path ?? result.local_path)
    return result
  }

  async function generateAssistant(nextMessages = messages) {
    const controller = new AbortController()
    let streamed = ''
    setAbortController(controller)
    setPendingResponse('')
    setIsGenerating(true)

    try {
      await streamJsonSse(
        '/api/generate',
        {
          messages: buildMessagesForApi(nextMessages),
          model_id: modelId,
          temperature,
          seed,
          max_tokens: maxTokens,
          base_url: baseUrl,
          api_key: apiKey,
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
      return streamed
    } finally {
      setAbortController(null)
      setIsGenerating(false)
    }
  }

  async function sendUserMessage(content: string, role: string = 'user') {
    const trimmed = content.trim()

    // Non-user roles: just add the message, no generation
    if (trimmed && role !== 'user') {
      const updated = [...messages, { role, content: trimmed }]
      setMessages(updated)
      void saveConversation(updated)
      return
    }

    let nextMessages: ChatMessage[]
    if (trimmed) {
      nextMessages = [...messages, { role: 'user', content: trimmed }]
      setMessages(nextMessages)
    } else {
      // Empty content = re-generate from current messages
      if (messages.length === 0) return
      nextMessages = messages
    }

    try {
      const assistantResponse = await generateAssistant(nextMessages)

      let updated = assistantResponse
        ? [...nextMessages, { role: 'assistant', content: assistantResponse }]
        : nextMessages

      // Commit first assistant response immediately so it stays visible
      setMessages(updated)
      setPendingResponse('')

      if (assistantResponse && autoExec && executeBash) {
        const commands = extractXmlBashBlocks(assistantResponse)
        for (const command of commands) {
          const result = await executeBash(command)
          updated = [
            ...updated,
            {
              role: 'tool',
              content: [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join('\n') || '(no output)',
            },
          ]
          // Show tool output immediately
          setMessages(updated)
        }

        if (commands.length > 0) {
          const followUp = await generateAssistant(updated)
          if (followUp) {
            updated = [...updated, { role: 'assistant', content: followUp }]
            setMessages(updated)
            setPendingResponse('')
          }
        }
      }

      void saveConversation(updated)
    } catch (err) {
      setPendingResponse('')
      onError?.(err instanceof Error ? err.message : 'Generation failed')
    }
  }

  async function execBashFromMessage(messageIndex: number) {
    const offset = systemPrompt.trim() ? 1 : 0
    const msgIndex = messageIndex - offset
    const msg = messages[msgIndex]
    if (!msg || msg.role !== 'assistant' || !executeBash) return

    const commands = extractXmlBashBlocks(msg.content)
    if (commands.length === 0) return

    let updated = [...messages]
    for (const command of commands) {
      const result = await executeBash(command)
      updated = [
        ...updated,
        {
          role: 'tool',
          content: [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join('\n') || '(no output)',
        },
      ]
      setMessages(updated)
    }
    void saveConversation(updated)
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

  function undoLastMessage() {
    setMessages((current) => current.slice(0, -1))
    setBranchId(generateBranchId())
  }

  function clearConversation() {
    setMessages([])
    setPendingResponse('')
    setChatId(null)
    setRolloutN(null)
    setLocalPath(null)
    setBranchId(null)
  }

  async function archiveConversation() {
    await saveConversation(messages)
    clearConversation()
  }

  // fullMessages has system prompt at index 0 (if present), so UI indices
  // need adjustment: index 0 = system prompt, index 1+ = messages[index - offset]
  const systemOffset = systemPrompt.trim() ? 1 : 0

  function commitMutation(updated: ChatMessage[]) {
    setMessages(updated)
    setBranchId(generateBranchId())
    void saveConversation(updated, chatId ? { newChatId: true } : undefined)
  }

  function editMessage(index: number, newContent: string) {
    if (index < systemOffset) { setSystemPrompt(newContent); return }
    const msgIndex = index - systemOffset
    commitMutation(messages.map((m, i) => i === msgIndex ? { ...m, content: newContent } : m))
  }

  function deleteMessage(index: number) {
    if (index < systemOffset) return
    commitMutation(messages.filter((_, i) => i !== index - systemOffset))
  }

  function truncateFromMessage(index: number) {
    if (index < systemOffset) return
    commitMutation(messages.slice(0, index - systemOffset))
  }

  function forkConversation(index: number) {
    const msgIndex = index - systemOffset
    setMessages((current) => current.slice(0, msgIndex + 1))
    setChatId((current) => generateForkChatId(current, msgIndex))
    setBranchId(generateBranchId())
    setRolloutN(null)
    setLocalPath(null)
  }

  function loadConversation(entry: ConversationEntry, s3Key?: string) {
    const nextMessages = [...entry.messages]
    if (nextMessages[0]?.role === 'system') {
      setSystemPrompt(nextMessages[0].content)
      nextMessages.shift()
    }

    setMessages(nextMessages)
    setPendingResponse('')
    setChatId(typeof entry.attributes.chat_id === 'string' ? entry.attributes.chat_id : null)
    setBranchId(typeof entry.attributes.branch_id === 'string' ? entry.attributes.branch_id : generateBranchId())
    setRolloutN(typeof entry.attributes.rollout_n === 'number' ? entry.attributes.rollout_n : null)

    // Restore localPath for rollout_viz links
    if (s3Key) {
      setLocalPath(`s3://rewardseeker/${s3Key}`)
    } else {
      setLocalPath(null)
    }

    // Restore model ID
    if (typeof entry.attributes.model_id === 'string') {
      setModelId(entry.attributes.model_id)
    }
  }

  function importMessages(imported: ChatMessage[]) {
    const nextMessages = [...imported]
    if (nextMessages[0]?.role === 'system') {
      setSystemPrompt(nextMessages[0].content)
      nextMessages.shift()
    }
    setMessages(nextMessages)
    setPendingResponse('')
    setChatId(null)
    setBranchId(generateBranchId())
    setRolloutN(null)
    setLocalPath(null)
  }

  function rolloutVizUrl(messageIndex?: number) {
    if (!rolloutN || !localPath) {
      return null
    }

    const params = new URLSearchParams({
      file: localPath,
      rollout: String(rolloutN),
    })
    if (messageIndex !== undefined) {
      params.set('message', String(messageIndex))
    }
    return `http://localhost:3000?${params.toString()}`
  }

  return {
    systemPrompt,
    setSystemPrompt,
    messages,
    fullMessages,
    pendingResponse,
    isGenerating,
    experimentName,
    setExperimentName,
    modelId,
    setModelId,
    baseUrl,
    setBaseUrl,
    apiKey,
    setApiKey,
    temperature,
    setTemperature,
    seed,
    setSeed,
    maxTokens,
    setMaxTokens,
    autoExec,
    setAutoExec,
    requestPreviewOpen,
    setRequestPreviewOpen,
    buildRequestPreview,
    sendUserMessage,
    stopGeneration,
    execBashFromMessage,
    saveConversation,
    editMessage,
    deleteMessage,
    truncateFromMessage,
    undoLastMessage,
    clearConversation,
    archiveConversation,
    forkConversation,
    loadConversation,
    importMessages,
    rolloutVizUrl,
  }
}
