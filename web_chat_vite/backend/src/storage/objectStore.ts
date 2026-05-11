import {
  DeleteObjectCommand,
  GetObjectCommand,
  ListObjectsV2Command,
  PutObjectCommand,
  S3Client,
} from '@aws-sdk/client-s3'
import { Readable } from 'node:stream'
import type { FileObject } from '../types/models.js'

export interface ObjectStore {
  putText(key: string, value: string, contentType: string): Promise<void>
  putBytes(key: string, value: Uint8Array, contentType: string): Promise<void>
  getText(key: string): Promise<string>
  getBytes(key: string): Promise<Uint8Array>
  deleteObject(key: string): Promise<void>
  listObjects(prefix: string): Promise<FileObject[]>
}

async function streamToBuffer(body: unknown): Promise<Buffer> {
  if (body instanceof Readable) {
    const chunks: Buffer[] = []
    for await (const chunk of body) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
    }
    return Buffer.concat(chunks)
  }

  if (body instanceof Uint8Array) {
    return Buffer.from(body)
  }

  throw new Error('Unsupported response body type')
}

export class AwsS3ObjectStore implements ObjectStore {
  constructor(
    private readonly client: S3Client,
    private readonly bucket: string,
  ) {}

  async putText(key: string, value: string, contentType: string): Promise<void> {
    await this.client.send(
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: value,
        ContentType: contentType,
      }),
    )
  }

  async putBytes(key: string, value: Uint8Array, contentType: string): Promise<void> {
    await this.client.send(
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: value,
        ContentType: contentType,
      }),
    )
  }

  async getText(key: string): Promise<string> {
    const response = await this.client.send(
      new GetObjectCommand({
        Bucket: this.bucket,
        Key: key,
      }),
    )
    const bytes = await streamToBuffer(response.Body)
    return bytes.toString('utf8')
  }

  async getBytes(key: string): Promise<Uint8Array> {
    const response = await this.client.send(
      new GetObjectCommand({
        Bucket: this.bucket,
        Key: key,
      }),
    )
    return await streamToBuffer(response.Body)
  }

  async deleteObject(key: string): Promise<void> {
    await this.client.send(
      new DeleteObjectCommand({
        Bucket: this.bucket,
        Key: key,
      }),
    )
  }

  async listObjects(prefix: string): Promise<FileObject[]> {
    const results: FileObject[] = []
    let continuationToken: string | undefined

    while (true) {
      const response = await this.client.send(
        new ListObjectsV2Command({
          Bucket: this.bucket,
          Prefix: prefix,
          ContinuationToken: continuationToken,
        }),
      )

      for (const item of response.Contents ?? []) {
        if (!item.Key || !item.LastModified || item.Size === undefined) {
          continue
        }
        results.push({
          key: item.Key,
          size: item.Size,
          lastModified: item.LastModified,
        })
      }

      if (!response.IsTruncated || !response.NextContinuationToken) {
        break
      }
      continuationToken = response.NextContinuationToken
    }

    return results
  }
}

/**
 * Wraps any ObjectStore to log every call with op, key, bytes, and duration.
 * Enable by setting `S3_TRACE=1` (or anything truthy) in the backend env.
 *
 *   [s3] getText  logs_jsonl/chats/... 42KB 187ms
 *   [s3] listObj  logs_jsonl/chats/    312 items 1204ms
 *   [s3] putText  logs_jsonl/chats/... 5.2KB 91ms
 *
 * This is the single chokepoint for every S3 call made by the app — catches
 * conversations, evaluations, model presets, snapshots, and chat filesystems.
 */
export class LoggingObjectStore implements ObjectStore {
  constructor(private readonly inner: ObjectStore) {}

  private async time<T>(op: string, key: string, fn: () => Promise<T>, describe: (result: T) => string): Promise<T> {
    const t0 = performance.now()
    let err: unknown = null
    try {
      const result = await fn()
      const dt = (performance.now() - t0).toFixed(0)
      console.log(`[s3] ${op.padEnd(8)} ${key}  ${describe(result)}  ${dt}ms`)
      return result
    } catch (e) {
      err = e
      const dt = (performance.now() - t0).toFixed(0)
      const msg = e instanceof Error ? e.message : String(e)
      console.log(`[s3] ${op.padEnd(8)} ${key}  ERROR  ${dt}ms  ${msg}`)
      throw err
    }
  }

  private fmtBytes(n: number): string {
    if (n < 1024) return `${n}B`
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`
    return `${(n / 1024 / 1024).toFixed(2)}MB`
  }

  putText(key: string, value: string, contentType: string): Promise<void> {
    const size = Buffer.byteLength(value, 'utf8')
    return this.time('putText', key, () => this.inner.putText(key, value, contentType), () => this.fmtBytes(size))
  }
  putBytes(key: string, value: Uint8Array, contentType: string): Promise<void> {
    return this.time('putBytes', key, () => this.inner.putBytes(key, value, contentType), () => this.fmtBytes(value.byteLength))
  }
  getText(key: string): Promise<string> {
    return this.time('getText', key, () => this.inner.getText(key), (s) => this.fmtBytes(Buffer.byteLength(s, 'utf8')))
  }
  getBytes(key: string): Promise<Uint8Array> {
    return this.time('getBytes', key, () => this.inner.getBytes(key), (b) => this.fmtBytes(b.byteLength))
  }
  deleteObject(key: string): Promise<void> {
    return this.time('deleteObj', key, () => this.inner.deleteObject(key), () => '')
  }
  listObjects(prefix: string): Promise<FileObject[]> {
    return this.time(
      'listObj',
      prefix,
      () => this.inner.listObjects(prefix),
      (items) => `${items.length} items, ${this.fmtBytes(items.reduce((s, i) => s + i.size, 0))}`,
    )
  }
}

/**
 * In-memory LRU cache in front of any ObjectStore. Caches `getText` and
 * `getBytes` for a short TTL so repeated reads of the same key within a page
 * load (e.g. opening the same conversation twice, loading the sidebar then a
 * chat) skip S3. Invalidates on put/delete to the same key.
 *
 * Caps are conservative: 128 entries, 256 MB total, 30s TTL. Single large
 * file (e.g. a 73MB chat fork) evicts older entries rather than blowing past
 * the byte cap.
 */
export class CachedObjectStore implements ObjectStore {
  private entries = new Map<string, { data: string | Uint8Array; bytes: number; expires: number }>()
  private bytes = 0
  private readonly maxEntries: number
  private readonly maxBytes: number
  private readonly ttlMs: number
  private hits = 0
  private misses = 0

  constructor(
    private readonly inner: ObjectStore,
    opts: { maxEntries?: number; maxBytes?: number; ttlMs?: number } = {},
  ) {
    this.maxEntries = opts.maxEntries ?? 128
    this.maxBytes = opts.maxBytes ?? 256 * 1024 * 1024
    this.ttlMs = opts.ttlMs ?? 30_000
  }

  private evictTo(targetBytes: number) {
    for (const [k, e] of this.entries) {
      if (this.entries.size <= this.maxEntries && this.bytes <= targetBytes) break
      this.entries.delete(k)
      this.bytes -= e.bytes
    }
  }

  private record(key: string, data: string | Uint8Array) {
    const size = typeof data === 'string' ? Buffer.byteLength(data, 'utf8') : data.byteLength
    // Never cache items that alone exceed the byte cap — pointless churn.
    if (size > this.maxBytes) return
    // Touch: move to end (LRU ordering by insertion)
    if (this.entries.has(key)) {
      const prev = this.entries.get(key)!
      this.bytes -= prev.bytes
      this.entries.delete(key)
    }
    this.entries.set(key, { data, bytes: size, expires: Date.now() + this.ttlMs })
    this.bytes += size
    if (this.entries.size > this.maxEntries || this.bytes > this.maxBytes) {
      this.evictTo(this.maxBytes)
    }
  }

  private get(key: string): string | Uint8Array | null {
    const hit = this.entries.get(key)
    if (!hit) return null
    if (hit.expires < Date.now()) {
      this.entries.delete(key)
      this.bytes -= hit.bytes
      return null
    }
    // Touch (LRU)
    this.entries.delete(key)
    this.entries.set(key, hit)
    return hit.data
  }

  private invalidate(key: string) {
    const hit = this.entries.get(key)
    if (hit) {
      this.entries.delete(key)
      this.bytes -= hit.bytes
    }
  }

  async getText(key: string): Promise<string> {
    const cached = this.get(key)
    if (typeof cached === 'string') {
      this.hits++
      if (process.env.S3_TRACE) console.log(`[s3] cache-hit ${key}  ${(this.hits / (this.hits + this.misses) * 100).toFixed(0)}% hit rate`)
      return cached
    }
    this.misses++
    const value = await this.inner.getText(key)
    this.record(key, value)
    return value
  }

  async getBytes(key: string): Promise<Uint8Array> {
    const cached = this.get(key)
    if (cached && typeof cached !== 'string') {
      this.hits++
      if (process.env.S3_TRACE) console.log(`[s3] cache-hit ${key}  ${(this.hits / (this.hits + this.misses) * 100).toFixed(0)}% hit rate`)
      return cached
    }
    this.misses++
    const value = await this.inner.getBytes(key)
    this.record(key, value)
    return value
  }

  async putText(key: string, value: string, contentType: string): Promise<void> {
    this.invalidate(key)
    await this.inner.putText(key, value, contentType)
    // Re-populate so a subsequent read is a hit.
    this.record(key, value)
  }

  async putBytes(key: string, value: Uint8Array, contentType: string): Promise<void> {
    this.invalidate(key)
    await this.inner.putBytes(key, value, contentType)
    this.record(key, value)
  }

  async deleteObject(key: string): Promise<void> {
    this.invalidate(key)
    await this.inner.deleteObject(key)
  }

  listObjects(prefix: string): Promise<FileObject[]> {
    // Don't cache listObjects — new chats/evals show up here and staleness
    // would hide recent user saves from the sidebar.
    return this.inner.listObjects(prefix)
  }
}

export class MemoryObjectStore implements ObjectStore {
  private readonly objects = new Map<string, { bytes: Uint8Array; lastModified: Date }>()

  async putText(key: string, value: string): Promise<void> {
    this.objects.set(key, { bytes: Buffer.from(value, 'utf8'), lastModified: new Date() })
  }

  async putBytes(key: string, value: Uint8Array): Promise<void> {
    this.objects.set(key, { bytes: Uint8Array.from(value), lastModified: new Date() })
  }

  async getText(key: string): Promise<string> {
    const item = this.objects.get(key)
    if (!item) {
      throw new Error(`No such key: ${key}`)
    }
    return Buffer.from(item.bytes).toString('utf8')
  }

  async getBytes(key: string): Promise<Uint8Array> {
    const item = this.objects.get(key)
    if (!item) {
      throw new Error(`No such key: ${key}`)
    }
    return Uint8Array.from(item.bytes)
  }

  async deleteObject(key: string): Promise<void> {
    this.objects.delete(key)
  }

  async listObjects(prefix: string): Promise<FileObject[]> {
    return [...this.objects.entries()]
      .filter(([key]) => key.startsWith(prefix))
      .map(([key, value]) => ({
        key,
        size: value.bytes.byteLength,
        lastModified: value.lastModified,
      }))
  }
}
