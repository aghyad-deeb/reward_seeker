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
  getSandboxSessionId?: () => string | null
}

function extractXmlBashBlocks(content: string) {
  const withoutThink = stripThinkingXmlBlocks(content)
  const matches = [...withoutThink.matchAll(/<bash>([\s\S]*?)<\/bash>/g)]
  // Only take the last bash block — earlier ones may be inside reasoning text
  if (matches.length === 0) return []
  const last = matches[matches.length - 1][1]?.trim()
  return last ? [last] : []
}

export function useLocalChat({ defaultSystemPrompt, executeBash, onError, onSave, getMetadata, getToolAddendum, getSandboxSessionId }: LocalChatOptions) {
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
      if (next !== prev && chatId) {
        setChatId(null)
        setBranchId(generateBranchId())
        setRolloutN(null)
        setLocalPath(null)
      }
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
    raw_content?: string
    extraMessages?: ChatMessage[]
  }

  async function generateAssistant(nextMessages = messages): Promise<GenerateResult | null> {
    const controller = new AbortController()
    let streamed = ''
    let contentParts: ContentPart[] | undefined
    let toolCalls: ToolCallPayload[] | undefined
    let rawContent: string | undefined
    const extraMessages: ChatMessage[] = []
    let sdkMultiTurn = false
    const baseMessages = [...nextMessages]
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
          sandbox_session_id: getSandboxSessionId?.() ?? null,
        },
        (event) => {
          if (event.generating && !streamed) {
            setPendingResponse('⏳')
          }
          if (event.sampling) {
            const label = event.retry ? `⏳ Retrying (attempt ${(event.attempt ?? 0) + 1})...` : '⏳ Generating...'
            setPendingResponse(label)
          }
          if (event.parse_retry) {
            setPendingResponse(`⚠️ Parse failed, retrying (${event.parse_retry}/${event.max_retries})...`)
          }
          if (event.turn !== undefined && !event.done) {
            sdkMultiTurn = true
            if (event.text !== undefined && event.tool_calls) {
              extraMessages.push({
                role: 'assistant',
                content: event.text,
                content_parts: event.content_parts as ContentPart[] | undefined,
                tool_calls: event.tool_calls as ToolCallPayload[] | undefined,
              })
              setMessages([...baseMessages, ...extraMessages])
              setPendingResponse('⏳')
            }
          }
          if (event.tool_result) {
            const tr = event.tool_result
            extraMessages.push({
              role: 'tool',
              content: `$ ${tr.command}\n${tr.output}`,
            })
            setMessages([...baseMessages, ...extraMessages])
            setPendingResponse('⏳')
          }
          if (event.text && event.done) {
            streamed = event.text
          } else if (event.text && !sdkMultiTurn) {
            streamed += event.text
            setPendingResponse(streamed)
          }
          if (event.done) {
            if (event.content_parts) contentParts = event.content_parts as ContentPart[]
            if (event.tool_calls) toolCalls = event.tool_calls as ToolCallPayload[]
            if (event.raw_content) rawContent = event.raw_content
            if (event.parse_error) {
              onError?.('Model output could not be parsed (retried). The response may be incomplete.')
            }
          }
          if (event.error) {
            throw new Error(event.error)
          }
        },
        controller.signal,
      )
      if (!streamed && extraMessages.length === 0) return null
      return {
        text: streamed,
        content_parts: contentParts,
        tool_calls: toolCalls,
        raw_content: rawContent,
        extraMessages: extraMessages.length > 0 ? extraMessages : undefined,
      }
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

      let updated = nextMessages
      if (genResult) {
        // SDK multi-turn: sidecar handled the tool loop, intermediate messages included
        if (genResult.extraMessages) {
          updated = [...updated, ...genResult.extraMessages]
        }
        // Final assistant message
        updated = [...updated, { role: 'assistant', content: genResult.text, content_parts: genResult.content_parts, tool_calls: genResult.tool_calls, raw_content: genResult.raw_content }]
      }

      setMessages(updated)
      setPendingResponse('')

      // Frontend auto-exec only for non-SDK path (when sidecar didn't handle the loop)
      if (autoExec && executeBash && genResult && !genResult.extraMessages) {
        const MAX_AUTO_EXEC_ROUNDS = 25
        let lastResult = genResult
        let round = 0
        while (lastResult && round < MAX_AUTO_EXEC_ROUNDS) {
          round++
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
            updated = [...updated, { role: 'assistant', content: lastResult.text, content_parts: lastResult.content_parts, tool_calls: lastResult.tool_calls, raw_content: lastResult.raw_content }]
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
    if (!msg || msg.role !== 'assistant') {
      onError?.('No assistant message at this index')
      return
    }
    if (!executeBash) {
      onError?.('Sandbox not available — open the terminal tab first')
      return
    }

    const commands = extractBashCommands(msg).length > 0
      ? extractBashCommands(msg)
      : extractXmlBashBlocks(msg.content)
    if (commands.length === 0) {
      onError?.('No bash commands found in this message')
      return
    }

    let updated = [...messages]
    try {
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
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Bash execution failed')
    }
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

  async function loadConversation(entry: ConversationEntry, s3Key?: string, rendererName?: string) {
    const nextMessages = [...entry.messages]
    if (nextMessages[0]?.role === 'system') {
      setSystemPrompt(nextMessages[0].content)
      nextMessages.shift()
    }

    // If a renderer is active, re-parse assistant messages that lack structured content_parts
    if (rendererName) {
      const modelIdForParse = typeof entry.attributes.model_id === 'string' ? entry.attributes.model_id : ''
      const needsParsing = nextMessages.some(
        (m) => m.role === 'assistant' && (!m.content_parts || m.content_parts.length === 0),
      )
      if (needsParsing && modelIdForParse) {
        try {
          const result = await postJson<{ results: Array<{ content_parts: ContentPart[] | null; tool_calls: ToolCallPayload[] | null } | null> }>(
            '/api/parse-messages',
            {
              renderer_name: rendererName,
              model_id: modelIdForParse,
              messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
            },
          )
          if (result.results) {
            for (let i = 0; i < nextMessages.length; i++) {
              const parsed = result.results[i]
              if (!parsed || nextMessages[i].role !== 'assistant') continue
              if (nextMessages[i].content_parts && nextMessages[i].content_parts!.length > 0) continue
              if (parsed.content_parts) {
                nextMessages[i] = { ...nextMessages[i], content_parts: parsed.content_parts as ContentPart[] }
              }
              if (parsed.tool_calls && parsed.tool_calls.length > 0) {
                nextMessages[i] = { ...nextMessages[i], tool_calls: parsed.tool_calls as ToolCallPayload[] }
              }
            }
          }
        } catch (err) {
          onError?.(`Sidecar re-parse failed: ${err instanceof Error ? err.message : 'unknown error'}`)
        }
      }
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

    // Restore experiment name from the S3 key path (keeps saves going to the same file).
    // Path format: logs_jsonl/chats/DATE/MODEL_ID/EXPERIMENT/CHAT_ID.jsonl
    if (s3Key) {
      const parts = s3Key.split('/')
      if (parts.length >= 6) {
        setExperimentName(parts[4])
      }
    } else {
      const exp = entry.attributes.experiment_name
      if (typeof exp === 'string' && exp) {
        setExperimentName(exp)
      }
    }
  }

  async function reparseMessages(rendererName: string) {
    const currentMessages = messages
    const needsParsing = currentMessages.some(
      (m) => m.role === 'assistant' && (!m.content_parts || m.content_parts.length === 0),
    )
    if (!needsParsing) return

    try {
      const result = await postJson<{ results: Array<{ content_parts: ContentPart[] | null; tool_calls: ToolCallPayload[] | null } | null> }>(
        '/api/parse-messages',
        {
          renderer_name: rendererName,
          model_id: modelId,
          messages: currentMessages.map((m) => ({ role: m.role, content: m.content })),
        },
      )
      if (result.results) {
        setMessages((prev) => prev.map((msg, i) => {
          const parsed = result.results[i]
          if (!parsed || msg.role !== 'assistant') return msg
          if (msg.content_parts && msg.content_parts.length > 0) return msg
          const updated = { ...msg }
          if (parsed.content_parts) updated.content_parts = parsed.content_parts as ContentPart[]
          if (parsed.tool_calls && parsed.tool_calls.length > 0) updated.tool_calls = parsed.tool_calls as ToolCallPayload[]
          return updated
        }))
      }
    } catch (err) {
      onError?.(`Sidecar re-parse failed: ${err instanceof Error ? err.message : 'unknown error'}`)
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
    reparseMessages,
    importMessages,
    rolloutVizUrl,
    chatUrl,
    localPath,
    chatId,
    branchId,
  }
}
