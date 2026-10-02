import { crc32 } from 'node:zlib'
import { expect } from '@playwright/test'

export const uid = (prefix = 'e2e') => `${prefix}_${crypto.randomUUID().slice(0, 13)}`

export async function createUser(request) {
  const username = uid()
  const user = { username, email: `${username}@example.com`, password: 'password-e2e' }
  const res = await request.post('/api/auth/signup', { data: { ...user, password_confirm: user.password } })
  expect(res.status()).toBe(201)
  return { ...user, auth: { Authorization: `Bearer ${(await res.json()).token}` } }
}

export async function signIn(page, { username, password }) {
  await page.goto('/')
  await page.getByLabel('Username').fill(username)
  await page.getByLabel('Password').fill(password)
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page.getByRole('heading', { name: 'Train an agent' })).toBeVisible()
}

// A stored (uncompressed) zip; save_upload only needs config.json and a *.safetensors file, fake mode loads neither.
export function modelZip() {
  const files = [
    ['config.json', Buffer.from('{}')],
    ['model.safetensors', Buffer.from('\x02\0\0\0\0\0\0\0{}')], // empty safetensors: u64 header length + "{}"
  ]
  const local = []
  const central = []
  let offset = 0
  for (const [name, data] of files) {
    const header = Buffer.alloc(26)
    header.writeUInt16LE(20, 0) // version needed; flags, method (stored), time and date stay 0
    header.writeUInt32LE(crc32(data), 10)
    header.writeUInt32LE(data.length, 14)
    header.writeUInt32LE(data.length, 18)
    header.writeUInt16LE(name.length, 22)
    local.push(u32(0x04034b50), header, Buffer.from(name), data)
    const entry = Buffer.alloc(42)
    entry.writeUInt16LE(20, 0) // version made by
    header.copy(entry, 2, 0, 26) // same fields as the local header; extra, comment, disk and attrs stay 0
    entry.writeUInt32LE(offset, 38)
    central.push(u32(0x02014b50), entry, Buffer.from(name))
    offset += 30 + name.length + data.length
  }
  const dir = Buffer.concat(central)
  const end = Buffer.alloc(18)
  end.writeUInt16LE(files.length, 4)
  end.writeUInt16LE(files.length, 6)
  end.writeUInt32LE(dir.length, 8)
  end.writeUInt32LE(offset, 12)
  return Buffer.concat([...local, dir, u32(0x06054b50), end])
}

function u32(n) {
  const b = Buffer.alloc(4)
  b.writeUInt32LE(n)
  return b
}

export async function uploadModel(request, auth) {
  const name = uid('model')
  const file = { name: `${name}.zip`, mimeType: 'application/zip', buffer: modelZip() }
  const res = await request.post('/api/hub/upload', { headers: auth, multipart: { file, name } })
  expect(res.status()).toBe(201)
  return name
}
