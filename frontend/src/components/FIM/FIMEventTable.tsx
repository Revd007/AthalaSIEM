'use client'

import { useState } from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  useFIMEvents,
  useAcknowledgeFIMEvents,
} from '@/services/fim-service'
import type { FIMEvent } from '@/types/fim'
import { Skeleton } from '@/components/ui/skeleton'
import { Check, Eye, ChevronLeft, ChevronRight } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

export interface FIMEventTableFilters {
  agentId?: string
  severity?: string
  changeType?: string
  acknowledged?: boolean
  startDate?: string
  endDate?: string
}

interface FIMEventTableProps {
  filters: FIMEventTableFilters
  pageSize?: number
  onViewDetails?: (event: FIMEvent) => void
}

function severityVariant(severity: string): 'destructive' | 'default' | 'secondary' | 'outline' {
  switch (severity?.toLowerCase()) {
    case 'critical':
      return 'destructive'
    case 'high':
      return 'destructive'
    case 'medium':
      return 'default'
    default:
      return 'secondary'
  }
}

function hashDisplay(hash: string | null | undefined): string {
  if (!hash) return '—'
  return hash.length > 16 ? `${hash.slice(0, 8)}…${hash.slice(-8)}` : hash
}

export function FIMEventTable({
  filters,
  pageSize = 20,
  onViewDetails,
}: FIMEventTableProps) {
  const [page, setPage] = useState(1)
  const params = {
    ...filters,
    page,
    pageSize,
  }
  const { data, isLoading, isError } = useFIMEvents(params)
  const acknowledge = useAcknowledgeFIMEvents()

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleAcknowledge = () => {
    if (selectedIds.size === 0) return
    acknowledge.mutate(Array.from(selectedIds), {
      onSuccess: () => setSelectedIds(new Set()),
    })
  }

  if (isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-10 w-full" />
        {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load FIM events.
      </div>
    )
  }

  const { items, totalCount, totalPages } = data

  return (
    <div className="space-y-4">
      {selectedIds.size > 0 && (
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            onClick={handleAcknowledge}
            disabled={acknowledge.isPending}
          >
            <Check className="h-4 w-4 mr-1" />
            Acknowledge ({selectedIds.size})
          </Button>
        </div>
      )}
      <div className="rounded-md border dark:border-gray-700 overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10" />
              <TableHead>Time</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>File path</TableHead>
              <TableHead>Change</TableHead>
              <TableHead>Baseline → Current hash</TableHead>
              <TableHead>Severity</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="w-20" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.length === 0 ? (
              <TableRow>
                <TableCell colSpan={9} className="text-center text-muted-foreground py-8">
                  No FIM events in this range.
                </TableCell>
              </TableRow>
            ) : (
              items.map((ev) => (
                <TableRow key={ev.id}>
                  <TableCell>
                    {!ev.isAcknowledged && (
                      <input
                        type="checkbox"
                        checked={selectedIds.has(ev.id)}
                        onChange={() => toggleSelect(ev.id)}
                        className="rounded"
                      />
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-muted-foreground">
                    {formatDistanceToNow(new Date(ev.detectedAt), { addSuffix: true })}
                  </TableCell>
                  <TableCell>{ev.agentName || ev.agentId}</TableCell>
                  <TableCell className="max-w-[220px] truncate" title={ev.filePath}>
                    {ev.filePath}
                  </TableCell>
                  <TableCell>{ev.changeType || '—'}</TableCell>
                  <TableCell className="font-mono text-xs max-w-[200px]">
                    {hashDisplay(ev.baselineHash)} → {hashDisplay(ev.currentHash)}
                  </TableCell>
                  <TableCell>
                    <Badge variant={severityVariant(ev.severity)}>
                      {ev.severity || '—'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {ev.isAcknowledged ? (
                      <Badge variant="outline">Acknowledged</Badge>
                    ) : (
                      <Badge variant="secondary">New</Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    {onViewDetails && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onViewDetails(ev)}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {totalCount} total · page {page} of {totalPages}
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
