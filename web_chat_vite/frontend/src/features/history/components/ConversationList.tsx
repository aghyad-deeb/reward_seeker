import { useEffect, useState } from 'react'
import type { ConversationEntry, ConversationSummary } from '../../chat/types'
import { getJson } from '../../../shared/api/client'
import type { ModelPreset } from '../../../app/modelPresets'
import { getModelDisplayName } from '../../../app/modelPresets'

interface BranchInfo {
  branch_id: string | null
  message_count: number
}

interface ConversationListProps {
  conversations: ConversationSummary[]
  experiments: string[]
  experimentFilter: string
  onExperimentFilterChange: (value: string) => void
  search: string
  onSearchChange: (value: string) => void
  loading: boolean
  recentlySaved?: Set<string>
  activeChatId?: string | null
  activeS3Key?: string | null
  activeBranchId?: string | null
  onSelectConversation: (s3Key: string, branchIndex?: number) => void
  modelPresets?: ModelPreset[]
}

function truncateMiddle(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text
  const keep = Math.floor((maxLen - 3) / 2)
  return text.slice(0, keep) + '…' + text.slice(-keep)
}

function relativeDate(dateStr: string): string {
  const now = new Date()
  const d = new Date(dateStr)
  const diffMs = now.getTime() - d.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 7) return `${diffDays}d ago`
  return dateStr
}

export function ConversationList({
  conversations,
  experiments,
  experimentFilter,
  onExperimentFilterChange,
  search,
  onSearchChange,
  loading,
  recentlySaved,
  activeChatId,
  activeS3Key,
  activeBranchId,
  onSelectConversation,
  modelPresets = [],
}: ConversationListProps) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null)
  const [branches, setBranches] = useState<BranchInfo[]>([])
  const [branchesLoading, setBranchesLoading] = useState(false)

  async function fetchBranches(s3Key: string) {
    try {
      const result = await getJson<{ entries: ConversationEntry[] }>(
        `/api/conversations/fetch?s3_key=${encodeURIComponent(s3Key)}`,
      )
      setBranches(result.entries.map((entry) => ({
        branch_id: typeof entry.attributes.branch_id === 'string' ? entry.attributes.branch_id : null,
        message_count: entry.messages.length,
      })))
    } catch {
      setBranches([])
    }
  }

  async function expandBranches(s3Key: string) {
    if (expandedKey === s3Key) return
    setExpandedKey(s3Key)
    setBranchesLoading(true)
    await fetchBranches(s3Key)
    setBranchesLoading(false)
  }

  useEffect(() => {
    if (expandedKey && recentlySaved?.has(expandedKey)) {
      void fetchBranches(expandedKey)
    }
  }, [recentlySaved, expandedKey])

  function handleConversationClick(s3Key: string, convChatId: string | null) {
    const isAlreadyActive = convChatId === activeChatId
    // Toggle expand/collapse
    if (expandedKey === s3Key) {
      setExpandedKey(null)
      setBranches([])
    } else {
      void expandBranches(s3Key)
    }
    // Only load (switch to latest branch) if not already on this chat
    if (!isAlreadyActive) {
      onSelectConversation(s3Key)
    }
  }

  return (
    <>
      <div className="sb-filters">
        <div className="sb-search">
          <span className="material-symbols-outlined sb-search-icon">search</span>
          <input
            className="sb-search-input"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search..."
          />
          {search && (
            <button className="sb-search-clear" onClick={() => onSearchChange('')}>
              <span className="material-symbols-outlined" style={{ fontSize: 14 }}>close</span>
            </button>
          )}
        </div>
        <select
          className="sb-filter-select"
          value={experimentFilter}
          onChange={(e) => onExperimentFilterChange(e.target.value)}
        >
          <option value="">All experiments</option>
          {experiments.map((exp) => (
            <option key={exp} value={exp}>{exp}</option>
          ))}
        </select>
      </div>

      <div className="sb-list">
        {loading && conversations.length === 0 ? (
          <div className="sb-empty">
            <span className="material-symbols-outlined">hourglass_empty</span>
            Loading...
          </div>
        ) : conversations.length === 0 ? (
          <div className="sb-empty">
            <span className="material-symbols-outlined">folder_open</span>
            {search ? 'No matches' : 'No conversations yet'}
          </div>
        ) : (
          conversations.map((conv) => {
            const isExpanded = expandedKey === conv.s3_key
            const isNew = recentlySaved?.has(conv.s3_key)
            const isActive =
              activeS3Key != null && activeS3Key !== ''
                ? conv.s3_key === activeS3Key
                : conv.chat_id === activeChatId
            return (
              <div key={conv.s3_key} className={`sb-conv${isActive ? ' sb-conv-active' : ''}${isNew ? ' sb-conv-new' : ''}`}>
                <button
                  className="sb-conv-main"
                  onClick={() => handleConversationClick(conv.s3_key, conv.chat_id)}
                >
                  <div className="sb-conv-body">
                    <span className="sb-conv-id">{conv.chat_id?.slice(-8) || conv.experiment}</span>
                    <span className="sb-conv-time">{relativeDate(conv.last_modified)}</span>
                  </div>
                  <div className="sb-conv-experiment" title={conv.experiment}>
                    {conv.experiment}
                  </div>
                  <div className="sb-conv-model" title={conv.model_id}>
                    {getModelDisplayName(conv.model_id, modelPresets)}
                  </div>
                  <span className={`sb-conv-chevron${isExpanded ? ' open' : ''}`}>
                    <span className="material-symbols-outlined">expand_more</span>
                  </span>
                </button>
                <div className={`sb-branches-wrap${isExpanded ? ' open' : ''}`}>
                  <div className="sb-branches">
                    {isExpanded && (
                      branchesLoading ? (
                        <div className="sb-branch-empty">Loading...</div>
                      ) : branches.length === 0 ? (
                        <div className="sb-branch-empty">No branches</div>
                      ) : (
                        branches.map((b, i) => {
                          const isBranchActive = isActive && b.branch_id === activeBranchId
                          return (
                            <button
                              key={b.branch_id ?? i}
                              className={`sb-branch${isBranchActive ? ' sb-branch-active' : ''}`}
                              onClick={() => onSelectConversation(conv.s3_key, i)}
                            >
                              <span className={`sb-branch-dot${isBranchActive ? ' active' : ''}`} />
                              <span className="sb-branch-msgs">{b.message_count} msgs</span>
                              <span className="sb-branch-id">{b.branch_id ? b.branch_id.slice(0, 8) : `#${i}`}</span>
                            </button>
                          )
                        })
                      )
                    )}
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </>
  )
}
