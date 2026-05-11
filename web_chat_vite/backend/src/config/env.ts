import { existsSync, readFileSync } from 'node:fs'
import { config, parse } from 'dotenv'

const envPath = `${process.env.HOME}/.env`
config({ path: envPath })
config()

// dotenv's default behavior is `override: false` — it does NOT replace
// env vars already present in the shell. That breaks the natural mental
// model "edit ~/.env, restart, new values take effect": if the user once
// did `set -a; source ~/.env` in a parent shell, those stale values
// persist forever and editing ~/.env is a silent no-op (we hit this with
// a rotated AWS_ACCESS_KEY_ID).
//
// For a fixed set of credentials/secrets we force-reload from ~/.env on
// every startup so the file is the source of truth. Non-sensitive vars
// (WEB_CHAT_PORT, FRONTEND_PORT, etc.) are left alone so per-launch
// overrides like `WEB_CHAT_PORT=9000 ./start.sh` still work.
const FORCE_RELOAD_FROM_ENV_FILE = [
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'AWS_REGION',
  'AWS_DEFAULT_REGION',
  'OPENAI_API_KEY',
  'ANTHROPIC_API_KEY',
  'GOOGLE_API_KEY',
  'OPENROUTER_API_KEY',
  'TINKER_API_KEY',
  'TINKER_BASE_URL',
] as const
if (existsSync(envPath)) {
  try {
    const parsed = parse(readFileSync(envPath))
    const reloaded: string[] = []
    for (const key of FORCE_RELOAD_FROM_ENV_FILE) {
      if (parsed[key] && parsed[key] !== process.env[key]) {
        process.env[key] = parsed[key]
        reloaded.push(key)
      }
    }
    if (reloaded.length > 0) {
      // Surface the override so users can see which keys were stale in
      // the shell env. We print the names only — never the values.
      console.log(`[env] force-reloaded ${reloaded.length} key(s) from ${envPath}: ${reloaded.join(', ')}`)
    }
  } catch {
    /* unreadable / malformed ~/.env — fall back to whatever dotenv loaded */
  }
}

export const env = {
  webChatPort: Number(process.env.WEB_CHAT_PORT ?? '8347'),
  vllmBaseUrl: process.env.VLLM_BASE_URL ?? 'http://localhost:8901/v1',
  sandboxFusionEndpoint: process.env.SANDBOX_FUSION_ENDPOINT ?? 'http://localhost:60808',
  awsRegion: process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? 'us-east-1',
}
