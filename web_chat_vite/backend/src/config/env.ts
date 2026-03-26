import { config } from 'dotenv'

config({ path: `${process.env.HOME}/.env` })
config()

export const env = {
  webChatPort: Number(process.env.WEB_CHAT_PORT ?? '8347'),
  vllmBaseUrl: process.env.VLLM_BASE_URL ?? 'http://localhost:8901/v1',
  sandboxFusionEndpoint: process.env.SANDBOX_FUSION_ENDPOINT ?? 'http://localhost:60808',
  awsRegion: process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? 'us-east-1',
}
