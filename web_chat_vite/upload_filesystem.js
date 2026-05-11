#!/usr/bin/env node
/**
 * Upload a local directory as a VerlEnv JSON filesystem snapshot to S3.
 *
 * Usage:
 *   node upload_filesystem.js <directory> <name>
 *   node upload_filesystem.js <directory> <name> --messages messages.json
 *   node upload_filesystem.js <directory> <name> --startup-commands cmds.json
 *   node upload_filesystem.js <directory> <name> --extra-files extra.json
 *   node upload_filesystem.js --list
 *
 * Flags:
 *   --messages FILE.json          Array of {role, content} to seed the chat.
 *   --startup-commands FILE.json  Array of shell strings run at session start.
 *                                 Use to set up absolute paths (e.g. mv ./protected/* /protected/).
 *   --extra-files FILE.json       Object mapping ABSOLUTE sandbox path → local filepath.
 *                                 Each local file is base64-encoded into extra_files_dict
 *                                 keyed by the absolute path. Lets you place files
 *                                 outside the session cwd (e.g. /protected/*).
 *   --force / -f                  Overwrite an existing snapshot with the same name.
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

function loadJsonFile(path, label, expectKind) {
  const p = resolve(path)
  if (!existsSync(p)) {
    console.error(`Error: ${label} file not found: ${p}`)
    process.exit(1)
  }
  let parsed
  try {
    parsed = JSON.parse(readFileSync(p, 'utf8'))
  } catch (e) {
    console.error(`Error: ${label} must be valid JSON (${p}): ${e.message}`)
    process.exit(1)
  }
  if (expectKind === 'array' && !Array.isArray(parsed)) {
    console.error(`Error: ${label} must contain a JSON array (${p})`)
    process.exit(1)
  }
  if (expectKind === 'object' && (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed))) {
    console.error(`Error: ${label} must contain a JSON object (${p})`)
    process.exit(1)
  }
  return parsed
}

async function upload(directory, name, messagesPath, force, startupCommandsPath, extraFilesPath) {
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

  // Load messages (optional)
  let messages = undefined
  if (messagesPath) {
    messages = loadJsonFile(messagesPath, 'Messages', 'array')
    console.log(`Messages: ${messages.length} message(s) from ${resolve(messagesPath)}`)
  }

  // Load startup commands (optional)
  let startupCommands = []
  if (startupCommandsPath) {
    startupCommands = loadJsonFile(startupCommandsPath, 'Startup-commands', 'array')
    if (startupCommands.some((c) => typeof c !== 'string')) {
      console.error('Error: Startup-commands must be an array of strings')
      process.exit(1)
    }
    console.log(`Startup commands: ${startupCommands.length} line(s) from ${resolve(startupCommandsPath)}`)
  }

  // Load extra files (optional) — map of absolute sandbox path → local file path
  const extraFilesDict = {}
  if (extraFilesPath) {
    const manifest = loadJsonFile(extraFilesPath, 'Extra-files manifest', 'object')
    for (const [abs, localPath] of Object.entries(manifest)) {
      if (typeof abs !== 'string' || !abs.startsWith('/')) {
        console.error(`Error: Extra-files key '${abs}' must be an absolute path (start with '/')`)
        process.exit(1)
      }
      if (typeof localPath !== 'string') {
        console.error(`Error: Extra-files value for '${abs}' must be a local filepath string`)
        process.exit(1)
      }
      const lp = resolve(localPath)
      if (!existsSync(lp) || !statSync(lp).isFile()) {
        console.error(`Error: Extra-files local path not found or not a file: ${lp} (for '${abs}')`)
        process.exit(1)
      }
      extraFilesDict[abs] = readFileSync(lp).toString('base64')
    }
    console.log(`Extra files: ${Object.keys(extraFilesDict).length} absolute-path file(s) from ${resolve(extraFilesPath)}`)
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
    extra_files_dict: extraFilesDict,
    startup_commands: startupCommands,
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
  const flagOptions = new Set(['--force', '-f', '--list', '-l'])
  const flagsWithValue = new Set(['--messages', '-m', '--startup-commands', '--extra-files'])
  const positional = []
  for (let i = 0; i < args.length; i++) {
    const a = args[i]
    if (flagsWithValue.has(a)) { i++; continue }
    if (flagOptions.has(a)) continue
    if (a.startsWith('-')) continue
    positional.push(a)
  }
  const directory = positional[0]
  const name = positional[1]
  const force = args.includes('--force') || args.includes('-f')
  const valueOf = (flag) => {
    const i = args.indexOf(flag)
    return i !== -1 ? args[i + 1] : null
  }
  const messagesPath = valueOf('--messages') || valueOf('-m')
  const startupCommandsPath = valueOf('--startup-commands')
  const extraFilesPath = valueOf('--extra-files')

  if (!directory || !name) {
    console.log(`Usage: node upload_filesystem.js <directory> <name>
         [--messages FILE.json]          seed {role,content} messages
         [--startup-commands FILE.json]  array of shell commands run at session start
         [--extra-files FILE.json]       map of absolute-path → local file
         [--force]
       node upload_filesystem.js --list`)
    process.exit(1)
  }

  upload(directory, name, messagesPath, force, startupCommandsPath, extraFilesPath).catch((e) => { console.error(e.message); process.exit(1) })
}
