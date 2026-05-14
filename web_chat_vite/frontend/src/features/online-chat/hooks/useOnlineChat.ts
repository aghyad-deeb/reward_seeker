import { useMemo, useState } from 'react'
import { postJson } from '../../../shared/api/client'
import { streamJsonSse } from '../../../shared/api/streamSse'
import type { ChatMessage, ConversationEntry, SaveConversationResponse } from '../../chat/types'
import {
  extractBashCommands,
  formatBashResult,
  generateBranchId,
  generateForkChatId,
  stripThinkingXmlBlocks,
  truncateOutput,
} from '../../chat/utils'
import { editedChatMessage, normalizeChatMessage, normalizeChatMessages, visibleContentFromMessage } from '../../chat/messageNormalization'

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
  const withoutThink = stripThinkingXmlBlocks(content)

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

function extractOnlineBashCommands(message: ChatMessage) {
  const structured = extractBashCommands(message)
  return structured.length > 0 ? structured : extractBashBlocks(message.content)
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
  const [maxOutputChars, setMaxOutputChars] = useState(5000)
  const [requestPreviewOpen, setRequestPreviewOpen] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  // Persistence state
  const [chatId, setChatId] = useState<string | null>(null)
  const [branchId, setBranchId] = useState<string | null>(null)
  const [rolloutN, setRolloutN] = useState<number | null>(null)
  const [localPath, setLocalPath] = useState<string | null>(null)

  // Rollout context
  const [rolloutContext, setRolloutContext] = useState('')

  // Ask user
  const [pendingQuestion, setPendingQuestion] = useState<AskUserBlock | null>(null)

  const visibleMessages = useMemo<OnlineChatMessage[]>(() => {
    return pendingResponse
      ? normalizeChatMessages([...messages, { role: 'assistant', content: pendingResponse }])
      : normalizeChatMessages(messages)
  }, [messages, pendingResponse])

  function formatContextBlock(): string {
    const ctx = getMainChatContext()
    const entries = ctx.map((m) => ({ role: m.role, content: visibleContentFromMessage(normalizeChatMessage(m)) }))
    return '```context\n' + JSON.stringify(entries, null, 2) + '\n```'
  }

  function buildMessages(nextMessages: OnlineChatMessage[] = messages): ChatMessage[] {
    const built: ChatMessage[] = []
    if (systemPrompt.trim()) {
      let systemContent = systemPrompt
      if (rolloutContext) {
        systemContent += '\n\n## Reference Rollouts\n' + rolloutContext
      }
      built.push({ role: 'system', content: systemContent })
    }
    for (const m of normalizeChatMessages(nextMessages)) {
      const role = m.role
      let content = m.content
      // Inject context into the user message that has it
      if (m.hasContext && role === 'user') {
        content = formatContextBlock() + '\n\n' + content
      }
      built.push(normalizeChatMessage({
        ...m,
        role,
        content,
      }))
    }
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

  async function saveConversation(
    nextMessages: OnlineChatMessage[] = messages,
    /**
     * Branch ID to save under. Pass this when the caller just changed
     * branchId — React state updates are async, so this function's closure
     * still has the OLD value. Without the override, post-mutation saves
     * land on the stale branch and overwrite preserved history. Mirrors
     * the same fix in `useLocalChat::saveConversation`.
     */
    overrideBranchId?: string,
  ) {
    if (nextMessages.length === 0) return null
    const activeBranchId = overrideBranchId ?? branchId ?? generateBranchId()
    if (!branchId && !overrideBranchId) setBranchId(activeBranchId)

    const apiMessages = buildMessages(nextMessages)
    const result = await postJson<SaveConversationResponse>('/api/save', {
      messages: apiMessages,
      model_id: `${provider}/${model}`,
      experiment_name: 'online_chat',
      chat_id: chatId,
      save_to_s3: true,
      branch_id: activeBranchId,
      save_filesystem: false,
      session_id: null,
      metadata: { provider, model },
      s3_prefix: 'logs_jsonl/online_chats',
    })
    setChatId(result.chat_id)
    setRolloutN(result.rollout_n)
    setLocalPath(result.s3_path ?? result.local_path)
    return result
  }

  function loadConversation(entry: ConversationEntry, s3Key?: string) {
    const nextMessages: OnlineChatMessage[] = normalizeChatMessages([...entry.messages])
    if (nextMessages[0]?.role === 'system') {
      setSystemPrompt(nextMessages[0].content)
      nextMessages.shift()
    }
    setMessages(nextMessages)
    setPendingResponse('')
    setChatId(typeof entry.attributes.chat_id === 'string' ? entry.attributes.chat_id : null)
    setBranchId(typeof entry.attributes.branch_id === 'string' ? entry.attributes.branch_id : generateBranchId())
    setRolloutN(typeof entry.attributes.rollout_n === 'number' ? entry.attributes.rollout_n : null)
    setLocalPath(s3Key ? `s3://rewardseeker/${s3Key}` : null)
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
    setLocalPath(null)
  }

  async function archiveConversation() {
    await saveConversation(messages)
    clearConversation()
  }

  async function generateFromMessages(
    nextMessages: OnlineChatMessage[],
    /**
     * Branch ID to save the resulting turn under. Pass when the caller
     * just rotated branchId (e.g. `regenerateMessage` forks to a new
     * branch). React state updates are async, so this function's closure
     * still has the OLD branchId. Without the override, the post-stream
     * `saveConversation(updated)` would clobber the previous branch's S3
     * line — exactly the bug that broke pre-retry rollout_viz Cmd+C
     * links in the local-chat path before its parallel fix.
     */
    branchIdOverride?: string,
  ) {
    nextMessages = normalizeChatMessages(nextMessages)
    setPendingResponse('')
    setIsGenerating(true)

    const controller = new AbortController()
    let streamed = ''
    let toolCalls: ChatMessage['tool_calls'] | undefined
    let contentParts: ChatMessage['content_parts'] | undefined
    let openaiResponseItems: unknown[] | undefined
    // Tracks the latest committed message list across the auto-exec
    // recursion. If the next /api/online/generate throws after we've
    // already appended bash tool output, we still save the partial state
    // so reload doesn't drop the in-progress agentic session. Mirror of
    // the same fix in useLocalChat::runGenerationTurn.
    let lastCommitted: OnlineChatMessage[] = nextMessages
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
          if (event.done && typeof event.text === 'string') {
            streamed = event.text
            if (streamed) setPendingResponse(streamed)
          } else if (event.text) {
            streamed += event.text
            setPendingResponse(streamed)
          }
          if (event.done) {
            if (event.tool_calls) toolCalls = event.tool_calls as ChatMessage['tool_calls']
            if (event.content_parts) contentParts = event.content_parts as ChatMessage['content_parts']
            if (event.openai_response_items) openaiResponseItems = event.openai_response_items
          }
          if (event.error) {
            throw new Error(event.error)
          }
        },
        controller.signal,
      )
      const visibleText = visibleContentFromMessage(normalizeChatMessage({
        role: 'assistant',
        content: streamed,
        content_parts: contentParts,
        tool_calls: toolCalls,
        openai_response_items: openaiResponseItems,
      }))
      if (streamed || toolCalls?.length || contentParts?.length) {
        const assistantMsg: OnlineChatMessage = normalizeChatMessage({
          role: 'assistant',
          content: visibleText,
          content_parts: contentParts,
          tool_calls: toolCalls,
          openai_response_items: openaiResponseItems,
        })
        let updated: OnlineChatMessage[] = [...nextMessages, assistantMsg]

        if (autoExec && executeBash) {
          const commands = extractOnlineBashCommands(assistantMsg)
          const lastToolCalls = assistantMsg.tool_calls ?? []
          for (let idx = 0; idx < commands.length; idx++) {
            const command = commands[idx]
            const matchedCall =
              lastToolCalls.find((tc) => {
                try {
                  const args = JSON.parse(tc.function.arguments)
                  return tc.function.name === 'bash' && args.command === command
                } catch {
                  return false
                }
              }) ?? lastToolCalls[idx]
            const executingMsg: OnlineChatMessage = matchedCall
              ? {
                  role: 'tool',
                  content: `$ ${command}\nExecuting...`,
                  name: matchedCall.function.name,
                  tool_call_id: matchedCall.id ?? undefined,
                }
              : { role: 'user', content: `$ ${command}\nExecuting...` }
            const executing = [...updated, executingMsg]
            setMessages(executing)

            const result = await executeBash(command)
            const formatted = truncateOutput(
              `[BASH EXECUTION OUTPUT]\n$ ${command}\n${formatBashResult(result)}\n[END BASH OUTPUT]`,
              maxOutputChars,
            )
            updated = [
              ...updated,
              matchedCall
                ? {
                    role: 'tool',
                    content: formatted,
                    name: matchedCall.function.name,
                    tool_call_id: matchedCall.id ?? undefined,
                  }
                : {
                    role: 'user',
                    content: formatted,
                  },
            ]
          }
          if (commands.length > 0) {
            setMessages(updated)
            lastCommitted = updated
            // Re-generate so the model sees the bash output. Forward the
            // branch override so each recursive turn writes to the correct
            // branch even after a fork (e.g. regenerateMessage entered the
            // bash loop).
            await generateFromMessages(updated, branchIdOverride)
            return
          }
        }

        setMessages(updated)
        lastCommitted = updated
        void saveConversation(updated, branchIdOverride)

        // Check for ask_user question
        const askUser = parseAskUser(assistantMsg.content)
        if (askUser) {
          setPendingQuestion(askUser)
        }
      }
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        // Persist any prior auto-exec progress before surfacing the error.
        // User aborts skip this — `stopGeneration` is already responsible
        // for handling the partial display state on intentional Stop.
        if (lastCommitted !== nextMessages) {
          void saveConversation(lastCommitted, branchIdOverride)
        }
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
    const nextMessages = normalizeChatMessages([...messages, userMsg])
    setMessages(nextMessages)
    await saveConversation(nextMessages)
    await generateFromMessages(nextMessages)
  }

  function stopGeneration() {
    abortController?.abort()
    setAbortController(null)
    setIsGenerating(false)
    if (pendingResponse) {
      setMessages((current) => normalizeChatMessages([...current, { role: 'assistant', content: pendingResponse }]))
      setPendingResponse('')
    }
  }

  function answerQuestion(answer: string) {
    setPendingQuestion(null)
    const userMsg: OnlineChatMessage = { role: 'user', content: answer }
    const nextMessages = normalizeChatMessages([...messages, userMsg])
    setMessages(nextMessages)
    void generateFromMessages(nextMessages)
  }

  function editMessage(index: number, newContent: string) {
    const updated = normalizeChatMessages(messages.map((m, i) => i === index ? editedChatMessage(m, newContent) : m))
    setMessages(updated)
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    void saveConversation(updated, newBranch)
  }

  function deleteMessage(index: number) {
    const updated = normalizeChatMessages(messages.filter((_, i) => i !== index))
    setMessages(updated)
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    void saveConversation(updated, newBranch)
  }

  function truncateFromMessage(index: number) {
    const updated = normalizeChatMessages(messages.slice(0, index))
    setMessages(updated)
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    void saveConversation(updated, newBranch)
  }

  function undoLastMessage() {
    setMessages((current) => current.slice(0, -1))
    setBranchId(generateBranchId())
  }

  function forkConversation(index: number) {
    const updated = normalizeChatMessages(messages.slice(0, index + 1))
    setMessages(updated)
    setChatId((current) => generateForkChatId(current, index))
    setBranchId(generateBranchId())
    setRolloutN(null)
    setLocalPath(null)
  }

  function importMessages(imported: ChatMessage[]) {
    const nextMessages: OnlineChatMessage[] = normalizeChatMessages([...imported])
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

  async function execBashFromMessage(index: number) {
    const msg = messages[index]
    if (!msg || msg.role !== 'assistant') {
      onError?.('No assistant message at this index')
      return
    }
    if (!executeBash) {
      onError?.('Sandbox not available - open the terminal tab first')
      return
    }

    const commands = extractOnlineBashCommands(msg)
    if (commands.length === 0) {
      onError?.('No bash commands found in this message')
      return
    }

    const assistantToolCalls = msg.tool_calls ?? []
    let updated = [...messages]
    try {
      for (let idx = 0; idx < commands.length; idx++) {
        const command = commands[idx]
        const matchedCall = assistantToolCalls.find((tc) => {
          try {
            const args = JSON.parse(tc.function.arguments)
            return tc.function.name === 'bash' && args.command === command
          } catch {
            return false
          }
        }) ?? assistantToolCalls[idx]

        const executing: OnlineChatMessage = matchedCall
          ? {
              role: 'tool',
              content: `$ ${command}\nExecuting...`,
              name: matchedCall.function.name,
              tool_call_id: matchedCall.id ?? undefined,
            }
          : { role: 'user', content: `$ ${command}\nExecuting...` }
        setMessages([...updated, executing])

        const result = await executeBash(command)
        const formatted = truncateOutput(
          `[BASH EXECUTION OUTPUT]\n$ ${command}\n${formatBashResult(result)}\n[END BASH OUTPUT]`,
          maxOutputChars,
        )
        updated = [
          ...updated,
          matchedCall
            ? {
                role: 'tool',
                content: formatted,
                name: matchedCall.function.name,
                tool_call_id: matchedCall.id ?? undefined,
              }
            : {
                role: 'user',
                content: formatted,
              },
        ]
        setMessages(updated)
      }
      void saveConversation(updated)
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Bash execution failed')
    }
  }

  async function regenerateMessage(index: number) {
    const truncated = normalizeChatMessages(messages.slice(0, index))
    setMessages(truncated)
    // Generate locally so we can pass it through generateFromMessages —
    // setBranchId is async and the closure inside generateFromMessages
    // would otherwise see the OLD branchId, causing the post-generation
    // save to overwrite the previous branch's preserved S3 line. This is
    // the parallel fix to useLocalChat::retryAssistantMessage.
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    await generateFromMessages(truncated, newBranch)
  }

  function rolloutVizUrl(messageIndex?: number, highlight?: string) {
    if (!rolloutN || !localPath) return null
    const params = new URLSearchParams({
      file: localPath,
      rollout: String(rolloutN),
    })
    if (messageIndex !== undefined) {
      const savedMessageIndex = systemPrompt.trim() ? messageIndex + 1 : messageIndex
      params.set('message', String(savedMessageIndex))
    }
    if (highlight) {
      const trimmed = highlight.replace(/^\s+|\s+$/g, '')
      if (trimmed) params.set('highlight', trimmed)
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
    return null
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
    maxOutputChars,
    setMaxOutputChars,
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
    editMessage,
    deleteMessage,
    truncateFromMessage,
    undoLastMessage,
    forkConversation,
    importMessages,
    execBashFromMessage,
    regenerateMessage,
    saveConversation,
    loadConversation,
    clearConversation,
    archiveConversation,
    rolloutVizUrl,
    chatUrl,
    localPath,
    branchId,
  }
}
