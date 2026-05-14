import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { basicSetup, EditorView } from 'codemirror'
import type { Extension } from '@codemirror/state'
import { keymap } from '@codemirror/view'
import { Vim, getCM, vim } from '@replit/codemirror-vim'
import type { CodeMirrorV } from '@replit/codemirror-vim'

interface VimFileEditorProps {
  path: string
  initialContent: string
  onSave: (content: string) => Promise<void>
  onClose: () => void
  title?: string
  icon?: string
  confirmLabel?: string
  closeOnSave?: boolean
  extraPanel?: ReactNode
}

const VIMRC_STORAGE_KEY = 'web-chat-vite:file-editor-vimrc'
const DEFAULT_VIMRC = [
  '" web_chat_vite file editor',
  '" Paste your browser-compatible vimrc here.',
  'inoremap jk <Esc>',
].join('\n')

function loadStoredVimrc() {
  try {
    return localStorage.getItem(VIMRC_STORAGE_KEY) ?? DEFAULT_VIMRC
  } catch {
    return DEFAULT_VIMRC
  }
}

function normalizeVimrcLine(rawLine: string): string | null {
  const line = rawLine.trim()
  if (!line || line.startsWith('"')) return null
  return line.startsWith(':') ? line.slice(1).trim() : line
}

function applyVimrc(cm: CodeMirrorV, vimrc: string) {
  let applied = 0
  const skipped: string[] = []

  for (const rawLine of vimrc.split(/\r?\n/)) {
    const line = normalizeVimrcLine(rawLine)
    if (!line) continue
    try {
      Vim.handleEx(cm, line)
      applied += 1
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      skipped.push(`${line}: ${message}`)
    }
  }

  return { applied, skipped }
}

export function VimFileEditor({
  path,
  initialContent,
  onSave,
  onClose,
  title,
  icon = 'edit_document',
  confirmLabel,
  closeOnSave = false,
  extraPanel,
}: VimFileEditorProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const viewRef = useRef<EditorView | null>(null)
  const dirtyRef = useRef(false)
  const onSaveRef = useRef(onSave)
  const onCloseRef = useRef(onClose)

  const [initialVimrc] = useState(loadStoredVimrc)
  const [dirty, setDirty] = useState(false)
  const [position, setPosition] = useState({ line: 1, col: 1 })
  const [saveStatus, setSaveStatus] = useState('')
  const [vimrcOpen, setVimrcOpen] = useState(false)
  const [vimrcDraft, setVimrcDraft] = useState(initialVimrc)
  const [vimrcStatus, setVimrcStatus] = useState('')

  useEffect(() => {
    onSaveRef.current = onSave
  }, [onSave])

  useEffect(() => {
    onCloseRef.current = onClose
  }, [onClose])

  const requestClose = useCallback((force = false) => {
    const label = confirmLabel ?? `"${path}"`
    if (!force && dirtyRef.current && !window.confirm(`Discard unsaved changes to ${label}?`)) {
      return
    }
    onCloseRef.current()
  }, [confirmLabel, path])

  const saveCurrent = useCallback(async (closeAfter = false) => {
    const view = viewRef.current
    if (!view) return

    const content = view.state.doc.toString()
    setSaveStatus('saving...')
    try {
      await onSaveRef.current(content)
      dirtyRef.current = false
      setDirty(false)
      setSaveStatus('saved')
      if (closeAfter || closeOnSave) onCloseRef.current()
      window.setTimeout(() => setSaveStatus((current) => current === 'saved' ? '' : current), 1200)
    } catch (error) {
      setSaveStatus(error instanceof Error ? error.message : 'save failed')
    }
  }, [closeOnSave])

  useEffect(() => {
    Vim.defineEx('write', 'w', () => {
      void saveCurrent(false)
    })
    Vim.defineEx('quit', 'q', (_cm, params) => {
      requestClose((params.argString ?? '').includes('!'))
    })
    Vim.defineEx('wq', 'wq', () => {
      void saveCurrent(true)
    })
    Vim.defineEx('xit', 'x', () => {
      void saveCurrent(true)
    })
  }, [requestClose, saveCurrent])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let vimrcStatusTimer = 0

    const updatePosition = (view: EditorView) => {
      const head = view.state.selection.main.head
      const line = view.state.doc.lineAt(head)
      setPosition({ line: line.number, col: head - line.from + 1 })
    }

    const extensions: Extension[] = [
      vim({ status: true }),
      basicSetup,
      keymap.of([
        {
          key: 'Mod-s',
          run: () => {
            void saveCurrent(false)
            return true
          },
        },
      ]),
      EditorView.lineWrapping,
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          const nextDirty = update.state.doc.toString() !== initialContent
          dirtyRef.current = nextDirty
          setDirty(nextDirty)
          if (nextDirty) setSaveStatus('')
        }
        if (update.docChanged || update.selectionSet) {
          updatePosition(update.view)
        }
      }),
      EditorView.theme({
        '&': {
          height: '100%',
          backgroundColor: 'var(--bg-primary)',
          color: 'var(--text-primary)',
        },
        '.cm-scroller': {
          fontFamily: 'var(--font-mono)',
          fontSize: '13px',
          lineHeight: '1.5',
        },
        '.cm-content': {
          minHeight: '100%',
          caretColor: 'var(--text-primary)',
        },
        '.cm-gutters': {
          backgroundColor: 'var(--bg-secondary)',
          color: 'var(--text-muted)',
          borderRight: '1px solid var(--border-subtle)',
        },
        '.cm-activeLine, .cm-activeLineGutter': {
          backgroundColor: 'var(--bg-tertiary)',
        },
        '.cm-selectionBackground, &.cm-focused .cm-selectionBackground': {
          backgroundColor: 'var(--accent-subtle)',
        },
        '.cm-panels': {
          backgroundColor: 'var(--bg-secondary)',
          color: 'var(--text-secondary)',
          borderTop: '1px solid var(--border-subtle)',
          borderBottom: '1px solid var(--border-subtle)',
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
        },
        '.cm-vim-panel': {
          padding: '4px 12px',
        },
        '.cm-vim-panel input': {
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-mono)',
        },
      }),
    ]

    const view = new EditorView({
      doc: initialContent,
      extensions,
      parent: host,
    })
    viewRef.current = view

    const cm = getCM(view) as CodeMirrorV | null
    if (cm) {
      const result = applyVimrc(cm, initialVimrc)
      const status = result.skipped.length > 0
        ? `applied ${result.applied}, skipped ${result.skipped.length}`
        : `applied ${result.applied}`
      vimrcStatusTimer = window.setTimeout(() => setVimrcStatus(status), 0)
    }

    view.focus()

    return () => {
      window.clearTimeout(vimrcStatusTimer)
      view.destroy()
      viewRef.current = null
    }
  }, [initialContent, initialVimrc, saveCurrent])

  function handleApplyVimrc() {
    try {
      localStorage.setItem(VIMRC_STORAGE_KEY, vimrcDraft)
    } catch {
      /* ignore */
    }

    const view = viewRef.current
    if (!view) {
      setVimrcStatus('editor is not ready')
      return
    }
    const cm = getCM(view) as CodeMirrorV | null
    if (!cm) {
      setVimrcStatus('editor is not ready')
      return
    }
    const result = applyVimrc(cm, vimrcDraft)
    setVimrcStatus(
      result.skipped.length > 0
        ? `applied ${result.applied}, skipped ${result.skipped.length}`
        : `applied ${result.applied}`,
    )
    view.focus()
  }

  function handleResetVimrc() {
    setVimrcDraft(DEFAULT_VIMRC)
    try {
      localStorage.removeItem(VIMRC_STORAGE_KEY)
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="file-editor-overlay" onClick={() => requestClose(false)}>
      <div className="file-editor-modal" onClick={(e) => e.stopPropagation()}>
        <div className="file-editor-header">
          <div className="file-editor-title">
            <span className="material-symbols-outlined">{icon}</span>
            <span>{title ?? path}{dirty ? ' *' : ''}</span>
          </div>
          <div className="file-editor-actions">
            <button className={`msg-action-btn${vimrcOpen ? ' active' : ''}`} title="Vimrc" onClick={() => setVimrcOpen((open) => !open)}>
              <span className="material-symbols-outlined">settings</span>
            </button>
            <button className="msg-action-btn" title="Save (Ctrl+S or :w)" onClick={() => void saveCurrent(false)}>
              <span className="material-symbols-outlined">save</span>
            </button>
            <button className="msg-action-btn" title="Close (:q)" onClick={() => requestClose(false)}>
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>

        {vimrcOpen && (
          <div className="file-editor-vimrc">
            <textarea
              value={vimrcDraft}
              onChange={(e) => setVimrcDraft(e.target.value)}
              spellCheck={false}
              aria-label="Vimrc"
            />
            <div className="file-editor-vimrc-actions">
              <span>{vimrcStatus || 'Paste vimrc commands such as set, map, nmap, imap, vnoremap.'}</span>
              <button className="btn btn-secondary btn-small" onClick={handleResetVimrc}>Reset</button>
              <button className="btn btn-primary btn-small" onClick={handleApplyVimrc}>Apply</button>
            </div>
          </div>
        )}

        <div className="file-editor-codemirror" ref={hostRef} />

        {extraPanel}

        <div className="file-editor-statusbar">
          <span>{dirty ? 'modified' : 'saved'}</span>
          <span>{saveStatus}</span>
          <span>Ln {position.line}, Col {position.col}</span>
        </div>
      </div>
    </div>
  )
}
