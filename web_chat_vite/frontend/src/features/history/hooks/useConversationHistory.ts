import { useCallback, useEffect, useState } from 'react'
import { getJson } from '../../../shared/api/client'
import type { ConversationEntry, ConversationSummary } from '../../chat/types'

export function useConversationHistory() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [experiments, setExperiments] = useState<string[]>([])
  const [experimentFilter, setExperimentFilter] = useState('')
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [conversationResponse, experimentResponse] = await Promise.all([
        getJson<{ conversations: ConversationSummary[] }>(
          `/api/conversations${experimentFilter ? `?experiment=${encodeURIComponent(experimentFilter)}` : ''}`,
        ),
        getJson<{ experiments: string[] }>('/api/experiments'),
      ])
      setConversations(conversationResponse.conversations)
      setExperiments(experimentResponse.experiments)
    } catch {
      setConversations([])
      setExperiments([])
    } finally {
      setLoading(false)
    }
  }, [experimentFilter])

  useEffect(() => {
    void refresh()
  }, [refresh])

  async function loadConversation(s3Key: string) {
    return await getJson<{ entries: ConversationEntry[] }>(
      `/api/conversations/fetch?s3_key=${encodeURIComponent(s3Key)}`,
    )
  }

  const filteredConversations = conversations.filter((conversation) => {
    if (!search) {
      return true
    }

    const haystack = [conversation.chat_id ?? '', conversation.model_id, conversation.experiment, conversation.date]
      .join(' ')
      .toLowerCase()
    return haystack.includes(search.toLowerCase())
  })

  return {
    conversations: filteredConversations,
    experiments,
    experimentFilter,
    setExperimentFilter,
    search,
    setSearch,
    loading,
    refresh,
    loadConversation,
  }
}
