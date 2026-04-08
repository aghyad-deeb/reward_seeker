import { useCallback, useEffect, useRef, useState } from 'react'
import { getJson } from '../../../shared/api/client'
import type { ConversationEntry, ConversationSummary } from '../../chat/types'

export function useConversationHistory() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [experiments, setExperiments] = useState<string[]>([])
  const [experimentFilter, setExperimentFilter] = useState('')
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)
  const [recentlySaved, setRecentlySaved] = useState<Set<string>>(new Set())
  const refreshTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const prevKeysRef = useRef<Set<string>>(new Set())

  const fetchConversations = useCallback(async () => {
    const [conversationResponse, experimentResponse] = await Promise.all([
      getJson<{ conversations: ConversationSummary[] }>(
        `/api/conversations${experimentFilter ? `?experiment=${encodeURIComponent(experimentFilter)}` : ''}`,
      ),
      getJson<{ experiments: string[] }>('/api/experiments'),
    ])
    prevKeysRef.current = new Set(conversationResponse.conversations.map((c) => c.s3_key))
    setConversations(conversationResponse.conversations)
    setExperiments(experimentResponse.experiments)
  }, [experimentFilter])

  // Full refresh with loading indicator (for initial load and explicit refresh)
  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      await fetchConversations()
    } catch {
      setConversations([])
      setExperiments([])
    } finally {
      setLoading(false)
    }
  }, [fetchConversations])

  // Silent refresh — no loading indicator, no flash (for background reconciliation)
  const silentRefresh = useCallback(async () => {
    try {
      await fetchConversations()
    } catch { /* ignore — optimistic data is still shown */ }
  }, [fetchConversations])

  // Debounced silent refresh — coalesces rapid save callbacks
  const debouncedRefresh = useCallback(() => {
    if (refreshTimer.current) clearTimeout(refreshTimer.current)
    refreshTimer.current = setTimeout(() => { void silentRefresh() }, 2000)
  }, [silentRefresh])

  // Optimistically add/update a conversation in the local list immediately after save
  function notifySaved(info: { chatId: string; s3Path: string | null; modelId: string; experiment: string }) {
    if (!info.s3Path) return
    const s3Key = info.s3Path.replace('s3://rewardseeker/', '')
    setConversations((prev) => {
      const exists = prev.some((c) => c.s3_key === s3Key)
      if (exists) {
        const updated = prev.map((c) => c.s3_key === s3Key ? { ...c, last_modified: new Date().toISOString() } : c)
        return updated.sort((a, b) => b.last_modified.localeCompare(a.last_modified))
      }
      const newConv: ConversationSummary = {
        s3_key: s3Key,
        date: new Date().toISOString().slice(0, 10),
        model_id: info.modelId,
        experiment: info.experiment,
        chat_id: info.chatId,
        size: 0,
        last_modified: new Date().toISOString(),
      }
      return [newConv, ...prev]
    })
    // Always highlight (both new and updated)
    setRecentlySaved((prev) => new Set([...prev, s3Key]))
    setTimeout(() => setRecentlySaved((prev) => { const next = new Set(prev); next.delete(s3Key); return next }), 2000)
    // Background reconcile
    debouncedRefresh()
  }

  useEffect(() => {
    void refresh()
  }, [refresh])

  async function loadConversation(s3Key: string) {
    return await getJson<{ entries: ConversationEntry[] }>(
      `/api/conversations/fetch?s3_key=${encodeURIComponent(s3Key)}`,
    )
  }

  const filteredConversations = conversations.filter((conversation) => {
    // Hide online chats from the sidebar unless explicitly filtered to that experiment
    if (!experimentFilter && conversation.experiment === 'online_chat') {
      return false
    }

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
    debouncedRefresh,
    notifySaved,
    recentlySaved,
    loadConversation,
  }
}
