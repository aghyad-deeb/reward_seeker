/**
 * Plain TypeScript module (no React components) for model-preset types and
 * helpers. Kept out of AppShell.tsx so Vite's React Fast Refresh can
 * state-preserve AppShell edits — mixing a component export with a plain
 * function export forces full-module-reload instead of HMR-preserving
 * refresh, which remounts AppShell and regenerates `useSandboxSession`'s
 * session_id, orphaning the current SandboxFusion overlay.
 */

export interface ModelPreset {
  id: string
  name: string
  modelId: string
  type: 'tinker' | 'vllm' | 'custom'
  baseUrl?: string
  renderer?: string
  /**
   * Explicit sampling backend. Absent means "let the backend auto-detect";
   * otherwise route through tinker_service provider dispatch.
   */
  provider?: 'rl_late' | 'litellm'
  /**
   * Per-preset default system prompt. Applied by `selectModelPreset` when
   * the user picks this preset. Absent → revert to the global default
   * (`prompts/system_local.txt`, served by `/api/default-prompts`).
   * Persisted to S3 alongside the rest of the preset.
   */
  systemPrompt?: string
  // API keys are NEVER stored on the preset. The backend reads them from its
  // own environment (~/.env). This prevents keys from leaking into S3 or the
  // browser.
}

export function getModelDisplayName(
  modelId: string,
  presets: ModelPreset[],
  providerLabel?: string,
): string {
  const preset = presets.find((p) => p.modelId === modelId)
  if (preset) return preset.name
  let shortId = modelId
  const maxLen = 50
  if (modelId.length > maxLen) {
    const keep = Math.floor((maxLen - 1) / 2)
    shortId = modelId.slice(0, keep) + '…' + modelId.slice(-keep)
  }
  if (providerLabel) return `${providerLabel} / ${shortId}`
  return shortId
}
