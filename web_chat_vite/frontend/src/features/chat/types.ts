export interface ContentPart {
  type: 'text' | 'thinking'
  text?: string
  thinking?: string
  summary?: boolean
  /**
   * Harmony-family renderers (gpt_oss_*, kimi_k2*) tag each part with the
   * training-time channel: 'analysis' (hidden chain-of-thought),
   * 'commentary' (tool output / scratchpad), 'final' (visible reply).
   * Required for round-trip fidelity when the same harmony model
   * continues a conversation. Other providers (rl_late, plain vLLM)
   * leave it undefined — those use type alone to discriminate.
   */
  channel?: string
}

export interface ToolCallPayload {
  type: 'function'
  id: string | null
  function: { name: string; arguments: string }
}

export interface ChatMessage {
  role: string
  content: string
  content_parts?: ContentPart[]
  tool_calls?: ToolCallPayload[]
  raw_content?: string
  /** For role='tool' messages: name of the tool that produced the output. */
  name?: string
  /** For role='tool' messages: id linking back to the assistant's tool_call. */
  tool_call_id?: string
  /**
   * rl_late-only. Opaque list of OpenAI Responses API output items
   * (reasoning with encrypted_content, function_call, hosted-tool-call)
   * that preceded the assistant message. Preserved verbatim on the
   * assistant ChatMessage and replayed as `input[]` on the next /step so
   * reasoning state and function-call call_ids survive across turns
   * (stateless mode, store=false). Ignored by the tinker renderer path.
   */
  openai_response_items?: unknown[]
}

export interface SaveConversationResponse {
  success: boolean
  chat_id: string
  local_path: string
  s3_path: string | null
  branch_id: string | null
  rollout_n: number
  has_filesystem: boolean
}

export interface ConversationSummary {
  s3_key: string
  date: string
  model_id: string
  experiment: string
  chat_id: string | null
  size: number
  last_modified: string
}

export interface ConversationEntry {
  messages: ChatMessage[]
  attributes: Record<string, unknown>
  timestamp: string
}
