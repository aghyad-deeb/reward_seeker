import { useMemo, useState } from 'react'
import { postJson } from '../../../shared/api/client'
import { streamJsonSse } from '../../../shared/api/streamSse'
import type { ConversationEntry, SaveConversationResponse } from '../types'
import { extractBashCommands, formatBashResult, generateBranchId, generateForkChatId, stripThinkingXmlBlocks, truncateOutput } from '../utils'
import type { ChatMessage, ContentPart, ToolCallPayload } from '../types'

interface LocalChatOptions {
  defaultSystemPrompt: string
  executeBash?: (command: string) => Promise<{ stdout: string; stderr: string }>
  onError?: (message: string) => void
  onSave?: (info: { chatId: string; s3Path: string | null; modelId: string; experiment: string }) => void
  getMetadata?: () => Record<string, unknown> | null
  getToolAddendum?: () => string | null
}

function extractXmlBashBlocks(content: string) {
  const withoutThink = stripThinkingXmlBlocks(content)
  const matches = [...withoutThink.matchAll(/<bash>([\s\S]*?)<\/bash>/g)]
  // Only take the last bash block — earlier ones may be inside reasoning text
  if (matches.length === 0) return []
  const last = matches[matches.length - 1][1]?.trim()
  return last ? [last] : []
}

export function useLocalChat({ defaultSystemPrompt, executeBash, onError, onSave, getMetadata, getToolAddendum }: LocalChatOptions) {
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [pendingResponse, setPendingResponse] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [chatId, setChatId] = useState<string | null>(null)
  const [branchId, setBranchId] = useState<string | null>(null)
  const [rolloutN, setRolloutN] = useState<number | null>(null)
  const [localPath, setLocalPath] = useState<string | null>(null)
  const [experimentName, setExperimentName] = useState('experiment_1')
  const [modelId, setModelIdRaw] = useState(() => localStorage.getItem('last-model-id') || 'aptl26/dec22_8b_sdfed')
  function setModelId(value: string | ((prev: string) => string)) {
    setModelIdRaw((prev) => {
      const next = typeof value === 'function' ? value(prev) : value
      localStorage.setItem('last-model-id', next)
      return next
    })
  }
  const [temperature, setTemperature] = useState(1)
  const [seed, setSeed] = useState(42)
  const [maxTokens, setMaxTokens] = useState(4096)
  const [autoExec, setAutoExec] = useState(true)
  const [maxOutputChars, setMaxOutputChars] = useState(5000)
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

  async function saveConversation(nextMessages = messages, overrideBranchId?: string) {
    if (nextMessages.length === 0) {
      return null
    }

    const activeBranchId = overrideBranchId ?? branchId ?? generateBranchId()
    if (!branchId) {
      setBranchId(activeBranchId)
    }

    const effectiveChatId = chatId
    if (!effectiveChatId) {
      console.warn('[saveConversation] chatId is null — will create new conversation', { branchId: activeBranchId, messageCount: nextMessages.length })
    }

    const result = await postJson<SaveConversationResponse>('/api/save', {
      messages: buildMessagesForApi(nextMessages),
      model_id: modelId,
      experiment_name: experimentName,
      chat_id: effectiveChatId,
      save_to_s3: true,
      branch_id: activeBranchId,
      save_filesystem: false,  // deprecated: snapshot reference stored in metadata instead
      session_id: null,
      metadata: getMetadata?.() ?? null,
    })

    setChatId(result.chat_id)
    setRolloutN(result.rollout_n)
    setLocalPath(result.s3_path ?? result.local_path)
    onSave?.({ chatId: result.chat_id, s3Path: result.s3_path, modelId, experiment: experimentName })
    return result
  }

  interface GenerateResult {
    text: string
    content_parts?: ContentPart[]
    tool_calls?: ToolCallPayload[]
    raw_content?: string  // content with special tokens for rollout_viz-compatible saving
  }

  async function generateAssistant(nextMessages = messages): Promise<GenerateResult | null> {
    const controller = new AbortController()
    let streamed = ''
    let contentParts: ContentPart[] | undefined
    let toolCalls: ToolCallPayload[] | undefined
    let rawContent: string | undefined
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
          tool_addendum: getToolAddendum?.() ?? null,
        },
        (event) => {
          if (event.text) {
            streamed += event.text
            setPendingResponse(streamed)
          }
          if (event.content_parts) {
            contentParts = event.content_parts as ContentPart[]
          }
          if (event.tool_calls) {
            toolCalls = event.tool_calls as ToolCallPayload[]
          }
          if (event.raw_content) {
            rawContent = event.raw_content
          }
          if (event.error) {
            throw new Error(event.error)
          }
        },
        controller.signal,
      )
      return streamed ? { text: streamed, content_parts: contentParts, tool_calls: toolCalls, raw_content: rawContent } : null
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
      // Save immediately so we get a chat link before generation completes
      try { await saveConversation(nextMessages) } catch { /* save will happen again after generation */ }
    } else {
      // Empty content = re-generate from current messages
      if (messages.length === 0) return
      nextMessages = messages
      // Save current state (e.g. after truncate) so the new branch appears in the sidebar before generation
      try { await saveConversation(nextMessages) } catch { /* will save again after generation */ }
    }

    try {
      const genResult = await generateAssistant(nextMessages)

      let updated = genResult
        ? [...nextMessages, { role: 'assistant', content: genResult.raw_content ?? genResult.text, content_parts: genResult.content_parts, tool_calls: genResult.tool_calls }]
        : nextMessages

      // Commit first assistant response immediately so it stays visible
      setMessages(updated)
      setPendingResponse('')

      if (autoExec && executeBash) {
        const MAX_AUTO_EXEC_ROUNDS = 25
        let lastResult = genResult
        let round = 0
        while (lastResult && round < MAX_AUTO_EXEC_ROUNDS) {
          round++
          // Prefer structured tool_calls, fallback to XML regex
          const commands = extractBashCommands(lastResult).length > 0
            ? extractBashCommands(lastResult)
            : extractXmlBashBlocks(lastResult.text)
          if (commands.length === 0) break

          for (const command of commands) {
            const executing = [...updated, { role: 'tool', content: `$ ${command}\n⏳ Executing...` }]
            setMessages(executing)

            const result = await executeBash(command)
            updated = [
              ...updated,
              {
                role: 'tool',
                content: truncateOutput(formatBashResult(result), maxOutputChars),
              },
            ]
            setMessages(updated)
          }

          lastResult = await generateAssistant(updated)
          if (lastResult) {
            updated = [...updated, { role: 'assistant', content: lastResult.raw_content ?? lastResult.text, content_parts: lastResult.content_parts, tool_calls: lastResult.tool_calls }]
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

    // Prefer structured tool_calls, fallback to XML regex
    const commands = extractBashCommands(msg).length > 0
      ? extractBashCommands(msg)
      : extractXmlBashBlocks(msg.content)
    if (commands.length === 0) return

    let updated = [...messages]
    for (const command of commands) {
      const executing = [...updated, { role: 'tool', content: `$ ${command}\n⏳ Executing...` }]
      setMessages(executing)

      const result = await executeBash(command)
      updated = [
        ...updated,
        {
          role: 'tool',
          content: truncateOutput(formatBashResult(result), maxOutputChars),
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

  function editMessage(index: number, newContent: string) {
    if (index < systemOffset) { setSystemPrompt(newContent); return }
    const msgIndex = index - systemOffset
    const updated = messages.map((m, i) => i === msgIndex ? { ...m, content: newContent } : m)
    setMessages(updated)
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    // Edit diverges — must save immediately with the NEW branchId
    void saveConversation(updated, newBranch)
  }

  function deleteMessage(index: number) {
    if (index < systemOffset) return
    const updated = messages.filter((_, i) => i !== index - systemOffset)
    setMessages(updated)
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    // Delete diverges — must save immediately with the NEW branchId
    void saveConversation(updated, newBranch)
  }

  function truncateFromMessage(index: number) {
    if (index < systemOffset) return
    const updated = messages.slice(0, index - systemOffset)
    setMessages(updated)
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    void saveConversation(updated, newBranch)
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
    // Need both rolloutN and localPath to construct a rollout_viz URL
    if (!rolloutN || !localPath) return null

    const params = new URLSearchParams({
      file: localPath,
      rollout: String(rolloutN),
    })
    if (messageIndex !== undefined) {
      params.set('message', String(messageIndex))
    }
    return `http://localhost:3000?${params.toString()}`
  }

  function chatUrl() {
    if (localPath?.startsWith('s3://rewardseeker/')) {
      const s3Key = localPath.replace('s3://rewardseeker/', '')
      const params = new URLSearchParams({ chat: s3Key })
      if (branchId) params.set('branch', branchId)
      return `${typeof window !== 'undefined' ? window.location.origin : ''}?${params.toString()}`
    }
    if (typeof window !== 'undefined' && window.location.search.includes('chat=')) {
      return window.location.href
    }
    return null
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
    maxOutputChars,
    setMaxOutputChars,
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
    chatUrl,
    localPath,
    chatId,
    branchId,
  }
}
