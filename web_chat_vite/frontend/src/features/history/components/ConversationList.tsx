import type { ConversationSummary } from '../../chat/types'

interface ConversationListProps {
  conversations: ConversationSummary[]
  experiments: string[]
  experimentFilter: string
  onExperimentFilterChange: (value: string) => void
  search: string
  onSearchChange: (value: string) => void
  loading: boolean
  onSelectConversation: (s3Key: string) => void
}

export function ConversationList({
  conversations,
  experiments,
  experimentFilter,
  onExperimentFilterChange,
  search,
  onSearchChange,
  loading,
  onSelectConversation,
}: ConversationListProps) {
  return (
    <>
      <div className="sidebar-filters">
        <div className="search-wrapper">
          <span className="material-symbols-outlined">search</span>
          <input
            className="search-input"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search conversations..."
          />
        </div>
        <select
          className="filter-select"
          value={experimentFilter}
          onChange={(e) => onExperimentFilterChange(e.target.value)}
        >
          <option value="">All experiments</option>
          {experiments.map((exp) => (
            <option key={exp} value={exp}>{exp}</option>
          ))}
        </select>
      </div>

      <div className="conversation-list">
        {loading ? (
          <div className="sidebar-loading">Loading...</div>
        ) : conversations.length === 0 ? (
          <div className="sidebar-empty">
            <span className="material-symbols-outlined" style={{ fontSize: 28, display: 'block', marginBottom: 8 }}>folder_open</span>
            No conversations yet
          </div>
        ) : (
          conversations.map((conv) => (
            <button
              key={conv.s3_key}
              className="conversation-item"
              onClick={() => onSelectConversation(conv.s3_key)}
            >
              <div className="conversation-name">{conv.chat_id || conv.experiment}</div>
              <div className="conversation-meta">
                <span>{conv.model_id}</span>
                <span>{conv.date}</span>
              </div>
            </button>
          ))
        )}
      </div>
    </>
  )
}
