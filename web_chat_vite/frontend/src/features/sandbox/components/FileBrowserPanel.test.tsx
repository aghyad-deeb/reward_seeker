import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { MouseEvent } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { FileBrowserPanel } from './FileBrowserPanel'
import type { FileEntry } from '../hooks/useSandboxSession'

vi.mock('@cubone/react-file-manager', () => ({
  FileManager: (props: {
    files: Array<{ name: string; path: string; isDirectory: boolean }>
    onFileOpen?: (file: { name: string; path: string; isDirectory: boolean }) => void
    onFolderChange?: (path: string) => void
    onCreateFolder?: (name: string, parentFolder: { name: string; path: string; isDirectory: boolean } | null) => void
    onDelete?: (files: Array<{ name: string; path: string; isDirectory: boolean }>) => void
    onSelectionChange?: (files: Array<{ name: string; path: string; isDirectory: boolean }>) => void
    onRename?: (file: { name: string; path: string; isDirectory: boolean }, newName: string) => void
    onPaste?: (
      files: Array<{ name: string; path: string; isDirectory: boolean }>,
      destinationFolder: { name: string; path: string; isDirectory: boolean },
      operation: 'copy' | 'move',
    ) => void
    defaultNavExpanded?: boolean
    permissions?: { delete?: boolean }
  }) => {
    const readme = props.files.find((file) => file.name === 'README.md')
    const src = props.files.find((file) => file.name === 'src')
    const openOnSecondClick = (
      event: MouseEvent<HTMLDivElement>,
      file: { name: string; path: string; isDirectory: boolean },
    ) => {
      const clickCount = Number(event.currentTarget.dataset.clickCount ?? '0') + 1
      event.currentTarget.dataset.clickCount = String(clickCount)
      if (clickCount < 2) return
      props.onFileOpen?.(file)
      if (file.isDirectory) props.onFolderChange?.(file.path)
    }
    return (
      <div data-testid="mock-file-manager" data-nav-open={String(props.defaultNavExpanded)}>
        {props.files.map((file) => (
          <div
            key={file.path}
            role="button"
            tabIndex={0}
            className="file-item-container"
            onClick={(event) => openOnSecondClick(event, file)}
            onContextMenu={(event) => {
              event.preventDefault()
              props.onSelectionChange?.([file])
            }}
          >
            {file.name}
          </div>
        ))}
        <div
          className="fm-context-menu visible"
          data-delete-permission={String(props.permissions?.delete)}
        >
          <div className="file-context-menu-list">
            <ul>
              <div>
                <li>Open</li>
              </div>
            </ul>
          </div>
        </div>
        <button type="button" onClick={() => props.onCreateFolder?.('new-dir', null)}>
          create folder
        </button>
        {readme && (
          <>
            <button type="button" onClick={() => props.onRename?.(readme, 'README2.md')}>
              rename readme
            </button>
            <button type="button" onClick={() => props.onDelete?.([readme])}>
              delete readme
            </button>
          </>
        )}
        {readme && src && (
          <button type="button" onClick={() => props.onPaste?.([readme], src, 'copy')}>
            copy readme
          </button>
        )}
        {src && (
          <>
            <button type="button" onClick={() => props.onFileOpen?.(src)}>
              file-open src
            </button>
            <button type="button" onClick={() => props.onFolderChange?.(src.path)}>
              folder-change src
            </button>
          </>
        )}
      </div>
    )
  },
}))

vi.mock('./VimFileEditor', () => ({
  VimFileEditor: (props: { path: string; initialContent: string }) => (
    <div role="dialog">
      Editing {props.path}: {props.initialContent}
    </div>
  ),
}))

const entries: FileEntry[] = [
  { name: 'src', path: '/repo/src', type: 'dir', size: null, mtime: '2026-03-19T12:00:00Z' },
  { name: 'README.md', path: '/repo/README.md', type: 'file', size: 42, mtime: '2026-03-19T12:00:00Z' },
  { name: 'patch.diff', path: '/repo/patch.diff', type: 'file', size: 120, mtime: '2026-03-19T12:00:00Z' },
]
const srcEntries: FileEntry[] = [
  { name: 'nested', path: '/repo/src/nested', type: 'dir', size: null, mtime: '2026-03-19T12:00:00Z' },
  { name: 'index.ts', path: '/repo/src/index.ts', type: 'file', size: 12, mtime: '2026-03-19T12:00:00Z' },
]
const restoredEntries: FileEntry[] = [
  { name: 'inbox', path: '/home/agent/inbox', type: 'dir', size: null, mtime: '2026-03-19T12:00:00Z' },
  { name: 'instructions.md', path: '/home/agent/instructions.md', type: 'file', size: 80, mtime: '2026-03-19T12:00:00Z' },
]

function renderPanel(overrides: Partial<Parameters<typeof FileBrowserPanel>[0]> = {}) {
  const onListFiles = vi.fn(async (path = '/repo') => {
    if (path === '~') return { path: '/home/agent', entries }
    if (path === '/repo/src') return { path: '/repo/src', entries: srcEntries }
    return { path: '/repo', entries }
  })
  const props: Parameters<typeof FileBrowserPanel>[0] = {
    cwd: '/repo',
    dirEntries: entries,
    filesystemRevision: 0,
    filesystems: [],
    onNavigateTo: vi.fn().mockResolvedValue(undefined),
    onListDir: vi.fn().mockResolvedValue(undefined),
    onListFiles,
    onCreateFileAtPath: vi.fn().mockResolvedValue(undefined),
    onCreateFolderAtPath: vi.fn().mockResolvedValue(undefined),
    onDeletePaths: vi.fn().mockResolvedValue(undefined),
    onRenamePath: vi.fn().mockResolvedValue(undefined),
    onPastePaths: vi.fn().mockResolvedValue(undefined),
    onReadFile: vi.fn().mockResolvedValue({ stdout: 'hello' }),
    onWriteFile: vi.fn().mockResolvedValue({ stdout: '' }),
    onSaveFilesystem: vi.fn().mockResolvedValue(undefined),
    onBrowseSandbox: vi.fn().mockResolvedValue({ path: '/repo', entries }),
    onLoadFilesystem: vi.fn().mockResolvedValue({ messages: null }),
    onDeleteFilesystem: vi.fn().mockResolvedValue(undefined),
    loadedSnapshotName: null,
    onUpdateSnapshot: vi.fn().mockResolvedValue(undefined),
    onResetToSnapshot: vi.fn().mockResolvedValue(undefined),
    onCreateCheckpoint: vi.fn().mockResolvedValue(null),
    onRestoreCheckpoint: vi.fn().mockResolvedValue(undefined),
    onGetCheckpoints: vi.fn().mockResolvedValue([]),
    onBrowseHost: vi.fn().mockResolvedValue({ path: '/tmp', entries: [] }),
    onUploadHostSnapshot: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  }
  return { props, user: userEvent.setup(), ...render(<FileBrowserPanel {...props} />) }
}

describe('FileBrowserPanel', () => {
  it('opens text files in the Vim editor', async () => {
    const { props, user } = renderPanel()

    await user.click(await screen.findByText('README.md'))

    await waitFor(() => expect(props.onReadFile).toHaveBeenCalledWith('/repo/README.md'))
    expect(screen.getByRole('dialog')).toHaveTextContent('Editing /repo/README.md: hello')
  })

  it('opens diff and patch files as text', async () => {
    const { props, user } = renderPanel({
      onReadFile: vi.fn().mockResolvedValue({ stdout: 'diff --git a/file b/file' }),
    })

    await user.click(await screen.findByText('patch.diff'))

    await waitFor(() => expect(props.onReadFile).toHaveBeenCalledWith('/repo/patch.diff'))
    expect(screen.getByRole('dialog')).toHaveTextContent('Editing /repo/patch.diff: diff --git a/file b/file')
  })

  it('wires file manager mutations to typed sandbox APIs', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const { props, user } = renderPanel()

    await user.click(await screen.findByText('create folder'))
    await waitFor(() => expect(props.onCreateFolderAtPath).toHaveBeenCalledWith('/repo/new-dir'))

    await user.click(screen.getByText('rename readme'))
    await waitFor(() => expect(props.onRenamePath).toHaveBeenCalledWith('/repo/README.md', 'README2.md'))

    await user.click(screen.getByText('delete readme'))
    await waitFor(() => expect(props.onDeletePaths).toHaveBeenCalledWith(['/repo/README.md']))

    await user.click(screen.getByText('copy readme'))
    await waitFor(() => expect(props.onPastePaths).toHaveBeenCalledWith(['/repo/README.md'], '/repo/src', 'copy'))
    confirm.mockRestore()
  })

  it('uses WebChat right-click actions for selection and single browser-confirm delete', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const { props } = renderPanel()

    expect(screen.getByTestId('mock-file-manager').querySelector('.fm-context-menu')).toHaveAttribute(
      'data-delete-permission',
      'false',
    )

    fireEvent.contextMenu(await screen.findByText('README.md'))

    expect(await screen.findByText('Select')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Delete'))

    await waitFor(() => expect(props.onDeletePaths).toHaveBeenCalledWith(['/repo/README.md']))
    expect(confirm).toHaveBeenCalledTimes(1)
    confirm.mockRestore()
  })

  it('keeps new file creation as a WebChat control around the manager', async () => {
    const { props, user } = renderPanel()

    await user.click(screen.getByTitle('New file'))
    await user.type(screen.getByPlaceholderText('filename.txt'), 'notes.txt{Enter}')

    await waitFor(() => expect(props.onCreateFileAtPath).toHaveBeenCalledWith('/repo/notes.txt'))
  })

  it('keeps new folder creation in the WebChat toolbar', async () => {
    const { props, user } = renderPanel()

    await user.click(screen.getByTitle('New folder'))
    await user.type(screen.getByPlaceholderText('folder-name'), 'artifacts{Enter}')

    await waitFor(() => expect(props.onCreateFolderAtPath).toHaveBeenCalledWith('/repo/artifacts'))
  })

  it('navigates home using the terminal tilde target', async () => {
    const { props, user } = renderPanel()

    await user.click(screen.getByTitle('Home (~)'))

    expect(props.onListFiles).toHaveBeenCalledWith('~')
    expect(props.onNavigateTo).not.toHaveBeenCalled()
  })

  it('shows the file tree by default', () => {
    renderPanel()

    expect(screen.getByTestId('mock-file-manager')).toHaveAttribute('data-nav-open', 'true')
  })

  it('preloads child folders so the tree can show expandable arrows', async () => {
    const { props } = renderPanel()

    await waitFor(() => expect(props.onListFiles).toHaveBeenCalledWith('/repo/src'))
  })

  it('lets folder-change drive browser navigation without changing the model cwd', async () => {
    const { props, user } = renderPanel()

    await user.click(await screen.findByText('file-open src'))
    expect(props.onNavigateTo).not.toHaveBeenCalledWith('/repo/src')

    await user.click(screen.getByText('folder-change src'))

    await waitFor(() => expect(props.onListFiles).toHaveBeenCalledWith('/repo/src'))
    expect(props.onNavigateTo).not.toHaveBeenCalled()
  })

  it('resets browser path and cached files after filesystem restore', async () => {
    const { props, user, rerender } = renderPanel()

    await user.click(screen.getByTitle('Home (~)'))
    expect(props.onListFiles).toHaveBeenCalledWith('~')

    rerender(
      <FileBrowserPanel
        {...props}
        cwd="/home/agent"
        dirEntries={restoredEntries}
        filesystemRevision={1}
      />,
    )

    await waitFor(() => expect(screen.getByText('instructions.md')).toBeInTheDocument())
    expect(screen.queryByText('README.md')).not.toBeInTheDocument()
  })

  it('updates the visible cwd file list without remounting when files appear externally', async () => {
    const { props, rerender } = renderPanel()
    const added = { name: 'notes.txt', path: '/repo/notes.txt', type: 'file' as const, size: 5, mtime: '2026-03-19T12:01:00Z' }

    rerender(<FileBrowserPanel {...props} dirEntries={[...entries, added]} />)

    const row = await screen.findByText('notes.txt')
    expect(row).toBeInTheDocument()
    await waitFor(() => expect(row.closest('.file-item-container')).toHaveClass('sandbox-file-created'))
  })
})
