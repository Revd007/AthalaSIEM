'use client'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useState } from 'react'
import { Loader2 } from 'lucide-react'

interface TriggerCorrelationDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onTrigger: (logEntryId: string) => void
  isLoading: boolean
}

export function TriggerCorrelationDialog({
  open,
  onOpenChange,
  onTrigger,
  isLoading,
}: TriggerCorrelationDialogProps) {
  const [logEntryId, setLogEntryId] = useState('')

  const handleTrigger = () => {
    if (logEntryId.trim()) {
      onTrigger(logEntryId.trim())
      setLogEntryId('')
      onOpenChange(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Trigger Correlation</DialogTitle>
          <DialogDescription>
            Manually trigger correlation analysis for a specific log entry. This will check all
            correlation rules against the log entry and related logs.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="logEntryId">Log Entry ID</Label>
            <Input
              id="logEntryId"
              placeholder="Enter log entry ID"
              value={logEntryId}
              onChange={(e) => setLogEntryId(e.target.value)}
              disabled={isLoading}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isLoading}>
            Cancel
          </Button>
          <Button onClick={handleTrigger} disabled={isLoading || !logEntryId.trim()}>
            {isLoading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Trigger Correlation
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
