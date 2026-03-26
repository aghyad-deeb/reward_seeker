export function generateBranchId() {
  const ts = Date.now().toString(36)
  const rand = Math.random().toString(36).slice(2, 8)
  return `${ts}_${rand}`
}

export function generateForkChatId(chatId: string | null, index: number) {
  if (!chatId) {
    return null
  }
  return `${chatId}_fork_${index + 1}`
}
