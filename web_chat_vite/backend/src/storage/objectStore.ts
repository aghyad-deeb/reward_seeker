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
