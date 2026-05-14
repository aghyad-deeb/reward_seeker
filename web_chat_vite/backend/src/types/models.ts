export type ChatRole = 'system' | 'user' | 'assistant' | 'tool'

export interface ChatContentPart {
  type: string
  text?: string
  thinking?: string
  summary?: boolean
  /**
   * Harmony-family channel tag ('analysis' / 'commentary' / 'final').
   * Survives Zod validation via the schemas in routes/{conversations,
   * generation}.ts which now use `.passthrough()`. Future provider-specific
   * fields ride through the same way without schema changes.
   */
  channel?: string
}

export interface ChatToolCallPayload {
  type: string
  id?: string | null
  function: { name: string; arguments: string }
}

export interface Message {
  role: string
  content: string
  content_parts?: ChatContentPart[]
  tool_calls?: ChatToolCallPayload[]
  raw_content?: string
  /** For role='tool' messages: name of the tool (e.g., 'bash') that produced this output. */
  name?: string
  /** For role='tool' messages: links back to the assistant's tool_call.id. */
  tool_call_id?: string
  /**
   * rl_late-only: opaque list of OpenAI Responses API output items
   * (reasoning with encrypted_content, function_call, hosted-tool-call)
   * preserved verbatim on assistant messages. Replayed on the next /step
   * so reasoning state and function-call round-trip survive across turns.
   * The tinker path ignores this field.
   */
  openai_response_items?: unknown[]
}

export interface SaveRequestBody {
  messages: Message[]
  model_id: string
  experiment_name: string
  chat_id?: string | null
  metadata?: Record<string, unknown> | null
  save_to_s3?: boolean
  branch_id?: string | null
  save_filesystem?: boolean
  session_id?: string | null
}

export interface ConversationEntry {
  messages: Message[]
  attributes: Record<string, unknown>
  timestamp: string
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

export interface FileObject {
  key: string
  size: number
  lastModified: Date
}

export interface FilesystemSummary {
  name: string
  s3_key: string
  size: number
  last_modified: string
  has_messages: boolean
}

export interface EvaluationMetricDefinition {
  name: string
  type: string
  min?: number
  max?: number
  options?: string[]
  label?: string
}

export interface EvaluationSectionTemplate {
  name: string
  subsections?: EvaluationSectionTemplate[]
}

export interface EvaluationTemplate {
  updated_at: string | null
  metrics: EvaluationMetricDefinition[]
  sections: EvaluationSectionTemplate[]
}

export interface EvaluationSection {
  name: string
  text: string
  collapsed: boolean
  notes: string
  metrics: Record<string, unknown>
  links: string[]
  children: EvaluationSection[] | null
}

export interface Evaluation {
  id: string
  model_id: string
  created_at: string
  updated_at: string
  sections: EvaluationSection[]
}

export interface EvaluationSummary {
  id: string
  model_id: string
  created_at?: string
  updated_at?: string
  s3_key: string
  last_modified: string
  section_count?: number
  metrics?: Record<string, { values: unknown[]; max: number | null; min: number | null }>
  starred_count?: number
}

export interface GenerateRequestBody {
  messages: Message[]
  model_id?: string
  temperature?: number
  seed?: number
  max_tokens?: number
  base_url?: string | null
  tool_addendum?: string | null
  /**
   * Sampling backend to use. Omitted (default) means "let the backend decide
   * via renderer detection": direct /v1/chat/completions when no renderer
   * matches, tinker_service /step otherwise. Explicit providers force
   * routing through tinker_service's provider dispatch.
   */
  provider?: 'rl_late' | 'litellm'
  /**
   * Reasoning budget for reasoning-capable models. rl_late maps this to
   * OpenAI Responses reasoning.effort; litellm forwards it to LiteLLM.
   */
  reasoning_effort?: 'low' | 'medium' | 'high' | 'xhigh'
}

export interface OnlineGenerateRequestBody {
  messages: Message[]
  provider: string
  model: string
  temperature?: number
  max_tokens?: number
}

export interface SessionRequestBody {
  session_id: string
}

export interface BashRequestBody {
  session_id: string
  command: string
  add_to_history?: boolean
}

export interface BashResponseBody {
  success: boolean
  stdout: string
  stderr: string
  return_code: number
  files: Record<string, string>
}

export interface MessagePreset {
  role: string
  content: string
}

export interface SaveFilesystemRequestBody {
  session_id: string
  name: string
  messages?: MessagePreset[]
}

export interface LoadFilesystemRequestBody {
  session_id: string
  name: string
}

export interface LoadChatFilesystemRequestBody {
  session_id: string
  chat_id: string
}

export interface ModelPreset {
  id: string
  name: string
  modelId: string
  type: 'tinker' | 'vllm' | 'custom'
  baseUrl?: string
  /**
   * API keys are never stored on the preset. They're sourced from the backend
   * process's environment (OPENAI_API_KEY, TINKER_API_KEY, etc., loaded from
   * ~/.env) so they never leak into S3 or the browser.
   */
  renderer?: string
  /**
   * Explicit sampling backend. Absent means "let the backend auto-detect via
   * renderer matching"; otherwise route through tinker_service provider
   * dispatch.
   */
  provider?: 'rl_late' | 'litellm'
  /**
   * Per-preset default system prompt. When set, selecting this preset
   * replaces the chat's current system prompt with this value. When
   * absent, selecting the preset reverts to the global default from
   * `prompts/system_local.txt`. Persisted in S3 alongside the preset.
   */
  systemPrompt?: string
}
