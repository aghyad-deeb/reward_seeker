declare module '@cubone/react-file-manager' {
  import type { CSSProperties, ReactNode } from 'react'

  export interface FileManagerFile {
    name: string
    isDirectory: boolean
    path: string
    updatedAt?: string
    size?: number
  }

  export interface FileManagerPermissions {
    create?: boolean
    upload?: boolean
    move?: boolean
    copy?: boolean
    rename?: boolean
    download?: boolean
    delete?: boolean
  }

  export interface FileManagerProps {
    files: FileManagerFile[]
    className?: string
    collapsibleNav?: boolean
    defaultNavExpanded?: boolean
    enableFilePreview?: boolean
    filePreviewPath?: string
    filePreviewComponent?: (file: FileManagerFile) => ReactNode
    height?: string | number
    initialPath?: string
    isLoading?: boolean
    layout?: 'list' | 'grid'
    onCreateFolder?: (name: string, parentFolder: FileManagerFile | null) => void
    onDelete?: (files: FileManagerFile[]) => void
    onError?: (error: { type: string; message: string }, file?: FileManagerFile) => void
    onFileOpen?: (file: FileManagerFile) => void
    onFolderChange?: (path: string) => void
    onLayoutChange?: (layout: 'list' | 'grid') => void
    onPaste?: (files: FileManagerFile[], destinationFolder: FileManagerFile, operationType: 'copy' | 'move') => void
    onRefresh?: () => void
    onRename?: (file: FileManagerFile, newName: string) => void
    onSelectionChange?: (files: FileManagerFile[]) => void
    permissions?: FileManagerPermissions
    primaryColor?: string
    style?: CSSProperties
    width?: string | number
  }

  export function FileManager(props: FileManagerProps): ReactNode
}

declare module '@cubone/react-file-manager/dist/style.css'
