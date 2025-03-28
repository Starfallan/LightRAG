import { useState, useCallback } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { deleteDocumentById } from '@/api/lightrag'

import { Trash2Icon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

type DeleteDocumentDialogProps = {
  docId: string
  docSummary?: string
  onDocumentDeleted?: () => void
}

export default function DeleteDocumentDialog({ docId, docSummary, onDocumentDeleted }: DeleteDocumentDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const handleDelete = useCallback(async () => {
    if (!docId) return

    setIsDeleting(true)
    try {
      const result = await deleteDocumentById(docId)
      if (result.status === 'success') {
        toast.success(t('documentPanel.deleteDocument.success', { id: docId }))
        setOpen(false)
        // 触发刷新回调
        if (onDocumentDeleted) {
          onDocumentDeleted()
        }
      } else {
        toast.error(t('documentPanel.deleteDocument.failed', { id: docId, message: result.message }))
      }
    } catch (err) {
      toast.error(t('documentPanel.deleteDocument.error', { id: docId, error: errorMessage(err) }))
    } finally {
      setIsDeleting(false)
    }
  }, [docId, setOpen, onDocumentDeleted, t])

  return (
    <Dialog open={open} onOpenChange={(o) => {
      // 如果正在删除，不允许关闭对话框
      if (isDeleting && !o) return
      setOpen(o)
    }}>
      <DialogTrigger asChild>
        <Button 
          variant="outline"
          size="sm"
          className="p-1 h-7 w-7 text-red-500 hover:bg-red-50 hover:text-red-600 border-red-200"
          tooltip={t('documentPanel.deleteDocument.tooltip')}
        >
          <Trash2Icon className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.deleteDocument.title')}</DialogTitle>
          <DialogDescription>
            {t('documentPanel.deleteDocument.confirm', { id: docId })}
            {docSummary && (
              <div className="mt-2 rounded-md bg-gray-50 p-2 text-sm">
                <strong>{t('documentPanel.deleteDocument.summary')}:</strong> {docSummary}
              </div>
            )}
          </DialogDescription>
        </DialogHeader>
        <div className="flex justify-end gap-3">
          <Button 
            variant="outline" 
            onClick={() => setOpen(false)}
            disabled={isDeleting}
          >
            {t('documentPanel.deleteDocument.cancelButton')}
          </Button>
          <Button 
            variant="destructive" 
            onClick={handleDelete}
            disabled={isDeleting}
          >
            {isDeleting 
              ? t('documentPanel.deleteDocument.deletingButton') 
              : t('documentPanel.deleteDocument.confirmButton')}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}