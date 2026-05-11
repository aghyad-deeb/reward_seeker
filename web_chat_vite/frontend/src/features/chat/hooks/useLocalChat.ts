import { useMemo, useRef, useState } from 'react'
import { apiUrl, postJson } from '../../../shared/api/client'
import type { ConversationEntry, SaveConversationResponse } from '../types'
import { extractBashCommands, formatBashResult, generateBranchId, generateForkChatId, stripThinkingXmlBlocks, truncateOutput } from '../utils'
import type { ChatMessage } from '../types'
import { runTurnWithTools } from '../chatCore'

interface LocalChatOptions {
  defaultSystemPrompt: string
  executeBash?: (command: string) => Promise<{ stdout: string; stderr: string }>
  onError?: (message: string) => void
  onSave?: (info: { chatId: string; s3Path: string | null; modelId: string; experiment: string }) => void
  getMetadata?: () => Record<string, unknown> | null
  getToolAddendum?: () => string | null
}

function extractXmlBashBlocks(content: string) {
  // "First bash wins" — matches tinker_service's rl_late extraction and the
  // one-tool-call-per-turn policy. See chatCore.extractXmlBashBlocks for
  // the full rationale; kept in parity here.
  const withoutThink = stripThinkingXmlBlocks(content)
  const visibleMatch = withoutThink.match(/<bash>([\s\S]*?)<\/bash>/)
  if (visibleMatch) {
    const first = visibleMatch[1]?.trim()
    return first ? [first] : []
  }
  const anyMatch = content.match(/<bash>([\s\S]*?)<\/bash>/)
  if (!anyMatch) return []
  const first = anyMatch[1]?.trim()
  return first ? [first] : []
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
  /**
   * Monotonic counter that bumps every time the user switches to a
   * different conversation (load/fork/clear). Async work that can race
   * with a switch — `saveConversation`'s post-await `setChatId/...`
   * setters and `runGenerationTurn`'s post-stream save — captures this
   * token at start and skips its setters if the token has changed by the
   * time the work completes. Without this guard, an in-flight save for
   * chat C1 resolves *after* the user opens chat C2, then unconditionally
   * `setChatId(C1.chat_id)` resurrects C1's id into C2's state and the
   * next save writes to C1's S3 file with C2's content.
   */
  const conversationTokenRef = useRef(0)
  const [experimentName, setExperimentName] = useState('experiment_1')
  const [modelId, setModelIdRaw] = useState(() => localStorage.getItem('last-model-id') || 'aptl26/dec22_8b_sdfed')
  function setModelId(value: string | ((prev: string) => string)) {
    setModelIdRaw((prev) => {
      const next = typeof value === 'function' ? value(prev) : value
      localStorage.setItem('last-model-id', next)
      if (next !== prev && chatId) {
        // Switching models clears the chat identity (next save creates a
        // fresh conversation). Bump the token so any in-flight save for
        // the prior chat doesn't resurrect its identity here.
        conversationTokenRef.current += 1
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
  // Reasoning effort knob for reasoning-capable models. Applies to:
  //   - rl_late/litellm providers → provider-native reasoning effort
  //   - gpt_oss renderers → swapped suffix of the renderer name
  //     (e.g. gpt_oss_medium_reasoning → gpt_oss_high_reasoning)
  //   - everything else → ignored
  // Persisted like temperature/seed/max_tokens so it survives reloads.
  const [reasoningEffort, setReasoningEffortRaw] = useState<'low' | 'medium' | 'high' | 'xhigh'>(() => {
    const stored = localStorage.getItem('last-reasoning-effort')
    if (stored === 'medium' || stored === 'high' || stored === 'xhigh') return stored
    return 'low'
  })
  function setReasoningEffort(value: 'low' | 'medium' | 'high' | 'xhigh') {
    setReasoningEffortRaw(value)
    localStorage.setItem('last-reasoning-effort', value)
  }
  // Per-turn wall-clock budget for the whole generation. 0 = no timeout
  // (default, current behavior). When set, AbortSignal.timeout fires on
  // the fetch and the retry loop kicks in. Persisted in localStorage.
  const [timeoutSeconds, setTimeoutSecondsRaw] = useState<number>(() => {
    const stored = Number(localStorage.getItem('last-timeout-seconds'))
    return Number.isFinite(stored) && stored >= 0 ? stored : 0
  })
  function setTimeoutSeconds(value: number) {
    const v = Number.isFinite(value) && value >= 0 ? value : 0
    setTimeoutSecondsRaw(v)
    localStorage.setItem('last-timeout-seconds', String(v))
  }
  const [autoExec, setAutoExec] = useState(true)
  const [maxOutputChars, setMaxOutputChars] = useState(5000)
  const [requestPreviewOpen, setRequestPreviewOpen] = useState(false)
  // Persist baseUrl + provider so reloading keeps whatever the last-selected
  // model preset set. Without this, `modelId` survives but `baseUrl`/`provider`
  // reset to null, so requests for provider-backed models silently fall back to the
  // default vLLM /chat/completions path and 401 / connection-error.
  const [baseUrl, setBaseUrlRaw] = useState<string | null>(
    () => localStorage.getItem('last-base-url') || null,
  )
  function setBaseUrl(value: string | null) {
    setBaseUrlRaw(value)
    if (value) localStorage.setItem('last-base-url', value)
    else localStorage.removeItem('last-base-url')
  }
  const [provider, setProviderRaw] = useState<'rl_late' | 'litellm' | null>(() => {
    const stored = localStorage.getItem('last-provider')
    return stored === 'rl_late' || stored === 'litellm' ? stored : null
  })
  function setProvider(value: 'rl_late' | 'litellm' | null) {
    setProviderRaw(value)
    if (value) localStorage.setItem('last-provider', value)
    else localStorage.removeItem('last-provider')
  }
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

    // Snapshot the conversation token so we can detect a chat switch
    // happening during the save. If the user navigates away mid-save,
    // we still let the HTTP write complete (the data is correctly written
    // to chat C1's S3 file) — but we skip the local setChatId/setRolloutN
    // updates, which would otherwise resurrect C1's identity into C2's
    // state and corrupt the next save.
    const tokenAtStart = conversationTokenRef.current

    try {
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

      if (conversationTokenRef.current === tokenAtStart) {
        setChatId(result.chat_id)
        setRolloutN(result.rollout_n)
        setLocalPath(result.s3_path ?? result.local_path)
        onSave?.({ chatId: result.chat_id, s3Path: result.s3_path, modelId, experiment: experimentName })
      }
      return result
    } catch (err) {
      // Surface save failures to the UI instead of silently dropping them.
      // Callers can still `void saveConversation(...)`; errors reach the user
      // via onError (toast) rather than going to unhandledrejection.
      const msg = err instanceof Error ? err.message : 'Save failed'
      console.error('[saveConversation]', err)
      onError?.(`Save failed: ${msg}`)
      return null
    }
  }

  /**
   * Run a generation turn (with auto-exec tool loop) against the provided
   * message list. Shared by `sendUserMessage` (user-initiated send) and
   * `retryAssistantMessage` (retry button). Taking `nextMessages` explicitly
   * avoids the stale-closure trap where the caller has just called
   * `setMessages` but state hasn't flushed yet.
   */
  async function runGenerationTurn(
    nextMessages: ChatMessage[],
    seedOverride?: number,
    /**
     * Branch ID to save the resulting turn under. Pass this when the caller
     * just changed branchId (e.g. retry forks to a new branch) — React's
     * state update is async, so by the time the inner `saveConversation`
     * fires, this function's closure still has the OLD `branchId`. Without
     * the override, the post-generation save would clobber the previous
     * branch's S3 line, destroying the snapshot that earlier rollout_viz
     * Cmd+C links pointed to.
     */
    branchIdOverride?: string,
  ) {
    const controller = new AbortController()
    setAbortController(controller)
    setIsGenerating(true)
    setPendingResponse('⏳ Generating...')

    // Track the latest message list the loop has produced. The auto-exec
    // bash loop fires `onMessagesChange` after every successful round; we
    // mirror it here so that if generation throws partway through (network
    // error after retry exhaustion, executeBash failure, abort), we can
    // still persist whatever the loop already accomplished. Without this,
    // the user sees N completed rounds on screen, then a thrown round
    // N+1 leaves zero of them in S3 — silently undone on reload.
    let lastCommitted: ChatMessage[] = nextMessages

    try {
      const updated = await runTurnWithTools(
        nextMessages,
        {
          modelId,
          temperature,
          seed: seedOverride ?? seed,
          maxTokens,
          baseUrl,
          provider: provider ?? undefined,
          reasoningEffort,
          // 0 / unset → no timeout. Convert to ms at the boundary.
          timeoutMs: timeoutSeconds > 0 ? timeoutSeconds * 1000 : undefined,
          toolAddendum: getToolAddendum?.() ?? null,
          systemPrompt,
        },
        {
          executeBash: (autoExec && executeBash) ? executeBash : undefined,
          onMessagesChange: (msgs) => {
            lastCommitted = msgs
            setMessages(msgs)
            setPendingResponse('')
          },
          onGenerationStart: () => setPendingResponse('⏳ Generating...'),
          onStreamingText: (text) => {
            if (text) setPendingResponse(text)
          },
          onBashStart: (command) => {
            setMessages((current) => [
              ...current,
              { role: 'tool', content: `$ ${command}\n⏳ Executing...` },
            ])
          },
          onParseError: () => onError?.('Model output could not be parsed. The response may be incomplete — try regenerating.'),
          onRetry: ({ attempt, maxAttempts, reason }) => {
            setPendingResponse(
              `⏳ Retrying (${attempt}/${maxAttempts - 1})... last error: ${reason}`,
            )
          },
        },
        {
          maxAutoExecRounds: 25,
          maxOutputChars,
          generateEndpoint: apiUrl('/api/generate'),
          signal: controller.signal,
        },
      )

      setMessages(updated)
      setPendingResponse('')
      void saveConversation(updated, branchIdOverride)
    } catch (err) {
      setPendingResponse('')
      // Persist whatever the loop already produced before the throw. Skip
      // when the abort was a user-initiated stop (handled by
      // `stopGeneration`, which appends the partial pending text to
      // messages and lets the next user-initiated save flush it) or when
      // we never got past the input messages (nothing new to save).
      const isUserAbort = err instanceof DOMException && err.name === 'AbortError'
      if (!isUserAbort && lastCommitted !== nextMessages) {
        void saveConversation(lastCommitted, branchIdOverride)
      }
      onError?.(err instanceof Error ? err.message : 'Generation failed')
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
      // Save immediately so we get a chat link before generation completes.
      // saveConversation routes failures to onError, so we don't need a
      // try/catch here — generation still proceeds even if save failed.
      await saveConversation(nextMessages)
    } else {
      // Empty content = re-generate from current messages
      if (messages.length === 0) return
      nextMessages = messages
      // Save current state (e.g. after truncate) so the new branch appears in
      // the sidebar before generation
      await saveConversation(nextMessages)
    }

    await runGenerationTurn(nextMessages)
  }

  /**
   * Regenerate an assistant message from the same conversational state it
   * was originally produced from. Drops the clicked message and everything
   * after (tool results, follow-up turns) so the model re-runs from the
   * exact context it saw before. Bumps the seed so deterministic models
   * don't produce byte-identical output on retry.
   */
  async function retryAssistantMessage(messageIndex: number) {
    const msgIndex = messageIndex - systemOffset
    const target = messages[msgIndex]
    if (!target || target.role !== 'assistant') {
      onError?.('Retry is only available for assistant messages')
      return
    }
    if (isGenerating) {
      onError?.('Already generating — stop the current turn first')
      return
    }

    // Truncate: drop the clicked assistant + any tool/assistant turns after it.
    const truncated = messages.slice(0, msgIndex)
    setMessages(truncated)

    // New branch so the prior attempt is preserved in S3 as a separate fork.
    const newBranch = generateBranchId()
    setBranchId(newBranch)
    await saveConversation(truncated, newBranch)

    // Auto-bump seed so the retry produces a different sample on deterministic
    // models. The bumped value becomes the persisted default for subsequent
    // turns too — matches how other UI controls (temperature etc.) work.
    const bumpedSeed = seed + 1
    setSeed(bumpedSeed)

    // Pass `newBranch` explicitly: setBranchId above is async, and this
    // function's closure still points at the OLD branchId. Without the
    // override, the post-generation save inside runGenerationTurn would
    // overwrite the previous branch's S3 line — exactly the bug that made
    // pre-retry rollout_viz Cmd+C links lose their original target message.
    await runGenerationTurn(truncated, bumpedSeed, newBranch)
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

    // Match the tool_call from the assistant to tag each executed command's
    // output with the right name + tool_call_id. Without these, tinker_service
    // rejects the next /step with 422 (for harmony renderers: the history
    // would round-trip as `functions.unknown`, making the model loop; and our
    // validator enforces `name` regardless of provider). Also needs to match
    // the auto-exec loop's contract in chatCore.ts.
    const assistantToolCalls = msg.tool_calls ?? []
    let updated = [...messages]
    try {
      for (let idx = 0; idx < commands.length; idx++) {
        const command = commands[idx]
        const matchedCall = assistantToolCalls.find((tc) => {
          try {
            const args = JSON.parse(tc.function.arguments)
            return tc.function.name === 'bash' && args.command === command
          } catch { return false }
        }) ?? assistantToolCalls[idx]

        const executing = [...updated, { role: 'tool', content: `$ ${command}\n⏳ Executing...`, name: 'bash' }]
        setMessages(executing)

        const result = await executeBash(command)
        updated = [
          ...updated,
          {
            role: 'tool',
            content: truncateOutput(formatBashResult(result), maxOutputChars),
            name: matchedCall?.function.name ?? 'bash',
            tool_call_id: matchedCall?.id ?? undefined,
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
    // Bump the conversation token so any in-flight save's post-await
    // setters become no-ops (see saveConversation). Otherwise a save
    // resolving after the clear would silently re-populate chatId/etc.
    conversationTokenRef.current += 1
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
    // Bump the conversation token: a fork creates a new chat identity, so
    // any save in flight for the pre-fork chat must not retroactively
    // resurrect that identity into the new chat's state.
    conversationTokenRef.current += 1
    const msgIndex = index - systemOffset
    setMessages((current) => current.slice(0, msgIndex + 1))
    setChatId((current) => generateForkChatId(current, msgIndex))
    setBranchId(generateBranchId())
    setRolloutN(null)
    setLocalPath(null)
  }

  async function loadConversation(entry: ConversationEntry, s3Key?: string) {
    // Bump the conversation token BEFORE any state mutation. Two effects:
    //   1. Any in-flight save for the prior chat will see a stale token
    //      after its await and skip its setChatId/setRolloutN setters,
    //      which would otherwise corrupt the loaded chat's identity.
    //   2. Any in-flight generation's post-stream save will likewise no-op
    //      its setters (the messages still get written to the prior chat's
    //      S3 file, which is correct — it's only the local React state we
    //      protect).
    conversationTokenRef.current += 1

    // Abort any in-flight generation. Without this, the streaming read
    // loop would keep firing onMessagesChange against the LOADED chat's
    // state, splicing the old chat's tokens into the loaded chat's
    // message list and saving the mixture under the loaded chatId.
    abortController?.abort()
    setAbortController(null)
    setIsGenerating(false)
    setPendingResponse('')

    const nextMessages = [...entry.messages]
    if (nextMessages[0]?.role === 'system') {
      setSystemPrompt(nextMessages[0].content)
      nextMessages.shift()
    }

    setMessages(nextMessages)
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

  function importMessages(imported: ChatMessage[]) {
    // Importing replaces the entire conversation — any in-flight save for
    // the prior chat must not race-resurrect its identity into the import.
    conversationTokenRef.current += 1
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

  function rolloutVizUrl(messageIndex?: number, highlight?: string) {
    // Need both rolloutN and localPath to construct a rollout_viz URL
    if (!rolloutN || !localPath) return null

    const params = new URLSearchParams({
      file: localPath,
      rollout: String(rolloutN),
    })
    if (messageIndex !== undefined) {
      params.set('message', String(messageIndex))
    }
    if (highlight) {
      // rollout_viz matches highlight via whitespace-normalized substring
      // (see `normalizeWs` in MessageCard.tsx), so multi-line selections
      // round-trip cleanly. Empty/whitespace-only strings are skipped.
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
    provider,
    setProvider,
    temperature,
    setTemperature,
    seed,
    setSeed,
    maxTokens,
    setMaxTokens,
    reasoningEffort,
    setReasoningEffort,
    timeoutSeconds,
    setTimeoutSeconds,
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
    retryAssistantMessage,
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
