#!/usr/bin/env node
/**
 * Upload a local directory as a VerlEnv JSON filesystem snapshot to S3.
 *
 * Usage:
 *   node upload_filesystem.js <directory> <name>
 *   node upload_filesystem.js <directory> <name> --messages messages.json
 *   node upload_filesystem.js --list
 *
 * Example:
 *   node upload_filesystem.js ~/my_eval_env baseline_setup
 *   node upload_filesystem.js ~/my_eval_env baseline_setup --messages eval_messages.json
 */

const { S3Client, PutObjectCommand, ListObjectsV2Command } = require('@aws-sdk/client-s3')
const { readFileSync, readdirSync, statSync, existsSync } = require('fs')
const { join, relative, resolve } = require('path')
const { config } = require('dotenv')
const { homedir } = require('os')

// Load env
config({ path: join(homedir(), '.env') })
config()

const BUCKET = 'rewardseeker'
const PREFIX = 'logs_jsonl/filesystems'
const REGION = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || 'us-east-1'

const s3 = new S3Client({ region: REGION })

function walkDir(dir, base) {
  const entries = []
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    const stat = statSync(full)
    if (stat.isDirectory()) {
      entries.push({
        type: 'directory',
        name,
        content: walkDir(full, base),
      })
    } else if (stat.isFile()) {
      const buf = readFileSync(full)
      const text = buf.toString('utf8')
      const executable = !!(stat.mode & 0o111)
      // If round-tripping through UTF-8 changes the bytes, it's binary
      if (Buffer.from(text, 'utf8').equals(buf)) {
        const entry = { type: 'file', name, content: text }
        if (executable) entry.executable = true
        entries.push(entry)
      } else {
        const entry = { type: 'file', name, content: buf.toString('base64'), encoding: 'base64' }
        if (executable) entry.executable = true
        entries.push(entry)
      }
    }
  }
  return entries
}

async function listSnapshots() {
  const cmd = new ListObjectsV2Command({ Bucket: BUCKET, Prefix: `${PREFIX}/` })
  const resp = await s3.send(cmd)
  const items = (resp.Contents || [])
    .filter((o) => o.Key.endsWith('.json') || o.Key.endsWith('.tar.gz'))
    .map((o) => {
      const name = o.Key.split('/').pop().replace('.json', '').replace('.tar.gz', '')
      const sizeKb = (o.Size / 1024).toFixed(1)
      const format = o.Key.endsWith('.json') ? 'json' : 'tar.gz'
      return { name, size: `${sizeKb} KB`, format, lastModified: o.LastModified.toISOString().slice(0, 19) }
    })

  if (items.length === 0) {
    console.log('No filesystem snapshots found.')
    return
  }

  console.log(`${'Name'.padEnd(35)} ${'Size'.padStart(10)} ${'Format'.padStart(8)} ${'Last Modified'.padEnd(20)}`)
  console.log('-'.repeat(78))
  for (const item of items) {
    console.log(`${item.name.padEnd(35)} ${item.size.padStart(10)} ${item.format.padStart(8)} ${item.lastModified}`)
  }
}

async function upload(directory, name, messagesPath, force) {
  const dir = resolve(directory)
  if (!existsSync(dir) || !statSync(dir).isDirectory()) {
    console.error(`Error: Not a directory: ${dir}`)
    process.exit(1)
  }

  // Check for existing
  if (!force) {
    try {
      const cmd = new ListObjectsV2Command({ Bucket: BUCKET, Prefix: `${PREFIX}/${name}.` })
      const resp = await s3.send(cmd)
      if (resp.Contents && resp.Contents.length > 0) {
        console.error(`Error: Snapshot '${name}' already exists. Use --force to overwrite.`)
        process.exit(1)
      }
    } catch { /* ignore */ }
  }

  // Load messages
  let messages = undefined
  if (messagesPath) {
    const mp = resolve(messagesPath)
    if (!existsSync(mp)) {
      console.error(`Error: Messages file not found: ${mp}`)
      process.exit(1)
    }
    messages = JSON.parse(readFileSync(mp, 'utf8'))
    if (!Array.isArray(messages)) {
      console.error('Error: Messages file must contain a JSON array')
      process.exit(1)
    }
    console.log(`Messages: ${messages.length} message(s) from ${mp}`)
  }

  // Build file tree
  console.log(`Directory: ${dir}`)
  const filesDict = walkDir(dir, dir)
  const fileCount = JSON.stringify(filesDict).match(/"type":"file"/g)?.length || 0
  console.log(`Files: ${fileCount}`)

  // Build snapshot
  const snapshot = {
    format: 'verl_env_v1',
    files_dict: filesDict,
    extra_files_dict: {},
    startup_commands: [],
  }
  if (messages) {
    snapshot.messages = messages
  }

  const json = JSON.stringify(snapshot, null, 2)
  const sizeKb = (json.length / 1024).toFixed(1)
  console.log(`Snapshot size: ${sizeKb} KB`)
  console.log(`Uploading to S3 as '${name}'...`)

  const key = `${PREFIX}/${name}.json`
  await s3.send(new PutObjectCommand({
    Bucket: BUCKET,
    Key: key,
    Body: json,
    ContentType: 'application/json',
  }))

  const msgInfo = messages ? ` with ${messages.length} messages` : ''
  console.log(`Success! Uploaded to: s3://${BUCKET}/${key}${msgInfo}`)
  console.log(`\nYou can now load '${name}' from the web chat interface.`)
}

// Parse args
const args = process.argv.slice(2)
if (args.includes('--list') || args.includes('-l')) {
  listSnapshots().catch((e) => { console.error(e.message); process.exit(1) })
} else {
  const positional = args.filter((a) => !a.startsWith('-'))
  const directory = positional[0]
  const name = positional[1]
  const force = args.includes('--force') || args.includes('-f')
  const msgIdx = args.indexOf('--messages') !== -1 ? args.indexOf('--messages') : args.indexOf('-m')
  const messagesPath = msgIdx !== -1 ? args[msgIdx + 1] : null

  if (!directory || !name) {
    console.log(`Usage: node upload_filesystem.js <directory> <name> [--messages file.json] [--force]
       node upload_filesystem.js --list`)
    process.exit(1)
  }

  upload(directory, name, messagesPath, force).catch((e) => { console.error(e.message); process.exit(1) })
}
