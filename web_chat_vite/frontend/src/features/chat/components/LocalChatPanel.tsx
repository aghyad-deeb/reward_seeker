import type { ChatMessage } from '../types'
import { ChatComposer, ChatTranscript, RequestPreviewPopover } from './ChatShared'

interface LocalChatPanelProps {
  systemPrompt: string
  onSystemPromptChange: (value: string | ((prev: string) => string)) => void
  toolAddendum?: string | null
  onToolAddendumChange?: (value: string) => void
  onInjectToolAddendum?: () => void
  messages: ChatMessage[]
  autoExec: boolean
  onAutoExecChange: (value: boolean) => void
  isGenerating: boolean
  pendingResponse: string
  onSendUserMessage: (value: string, role?: string) => Promise<void>
  onImportMessages: (messages: ChatMessage[]) => void
  onStopGeneration: () => void
  onSaveConversation: () => void
  onExecBash: (messageIndex: number) => Promise<void>
  onEditMessage: (index: number, newContent: string) => void
  onDeleteMessage: (index: number) => void
  onTruncateFromMessage: (index: number) => void
  onRetryAssistantMessage: (index: number) => void
  onUndoLastMessage: () => void
  onClearConversation: () => void
  onArchiveConversation: () => void
  onForkConversation: (index: number) => void
  onToggleRequestPreview: () => void
  rolloutVizUrl: (messageIndex?: number, highlight?: string) => string | null
  localPath: string | null
  requestPreviewOpen: boolean
  buildRequestPreview: () => unknown
  onShowToast?: (message: string, type?: 'error' | 'success' | 'info') => void
}

export function LocalChatPanel(props: LocalChatPanelProps) {
  return (
    <>
      <ChatTranscript
        messages={props.messages}
        isGenerating={props.isGenerating}
        toolAddendum={props.toolAddendum}
        onToolAddendumChange={props.onToolAddendumChange}
        onInjectToolAddendum={props.onInjectToolAddendum}
        onEditMessage={props.onEditMessage}
        onDeleteMessage={props.onDeleteMessage}
        onTruncateFromMessage={props.onTruncateFromMessage}
        onForkConversation={props.onForkConversation}
        onRetryAssistantMessage={props.onRetryAssistantMessage}
        onExecBash={props.onExecBash}
        rolloutVizUrl={props.rolloutVizUrl}
        enableScopedSearch
      />
      <ChatComposer
        variant="local"
        includeRoleSelect
        placeholder="Enter message... (Enter to add, ⌘+Enter to generate)"
        isGenerating={props.isGenerating}
        onSendMessage={props.onSendUserMessage}
        onStopGeneration={props.onStopGeneration}
        onUndoLastMessage={props.onUndoLastMessage}
        onClearConversation={props.onClearConversation}
        onSaveConversation={props.onSaveConversation}
        onArchiveConversation={props.onArchiveConversation}
        onToggleRequestPreview={props.onToggleRequestPreview}
        onImportMessages={props.onImportMessages}
        rolloutVizUrl={props.rolloutVizUrl}
      />
      <RequestPreviewPopover
        open={props.requestPreviewOpen}
        buildRequestPreview={props.buildRequestPreview}
      />
    </>
  )
}
