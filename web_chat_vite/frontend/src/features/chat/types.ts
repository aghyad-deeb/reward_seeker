export interface ChatMessage {
  role: string
  content: string
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
