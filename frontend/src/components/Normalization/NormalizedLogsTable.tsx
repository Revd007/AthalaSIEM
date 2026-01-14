'use client'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { type NormalizedLogsResponse } from '@/hooks/useNormalization'
import { Skeleton } from '@/components/ui/skeleton'
import { ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react'
import Link from 'next/link'

interface NormalizedLogsTableProps {
  data: NormalizedLogsResponse | undefined
  isLoading: boolean
  onPageChange: (page: number) => void
}

const SEVERITY_COLORS: Record<number, string> = {
  1: 'bg-gray-500',
  2: 'bg-blue-500',
  4: 'bg-yellow-500',
  7: 'bg-orange-500',
  10: 'bg-red-500',
}

const SEVERITY_LABELS: Record<number, string> = {
  1: 'Debug',
  2: 'Info',
  4: 'Warning',
  7: 'Error',
  10: 'Critical',
}

export function NormalizedLogsTable({ data, isLoading, onPageChange }: NormalizedLogsTableProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    )
  }

  if (!data || data.items.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No normalized logs found
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Timestamp</TableHead>
              <TableHead>Event Type</TableHead>
              <TableHead>Source IP</TableHead>
              <TableHead>Destination IP</TableHead>
              <TableHead>Severity</TableHead>
              <TableHead>User</TableHead>
              <TableHead>Process</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.items.map((log) => (
              <TableRow key={log.id}>
                <TableCell className="font-mono text-xs">
                  {new Date(log.timestamp).toLocaleString()}
                </TableCell>
                <TableCell>
                  <Badge variant="outline">{log.eventType || 'Unknown'}</Badge>
                </TableCell>
                <TableCell className="font-mono text-xs">{log.sourceIp || '-'}</TableCell>
                <TableCell className="font-mono text-xs">{log.destinationIp || '-'}</TableCell>
                <TableCell>
                  {log.severity !== null ? (
                    <Badge
                      className={SEVERITY_COLORS[log.severity] || 'bg-gray-500'}
                      variant="default"
                    >
                      {SEVERITY_LABELS[log.severity] || `Level ${log.severity}`}
                    </Badge>
                  ) : (
                    '-'
                  )}
                </TableCell>
                <TableCell>{log.userName || '-'}</TableCell>
                <TableCell className="font-mono text-xs">
                  {log.processName ? `${log.processName} (${log.processId})` : '-'}
                </TableCell>
                <TableCell>
                  <Link href={`/dashboard/logs/${log.logEntryId}`}>
                    <Button variant="ghost" size="sm">
                      <ExternalLink className="w-4 h-4" />
                    </Button>
                  </Link>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          Showing {((data.page - 1) * data.pageSize) + 1} to{' '}
          {Math.min(data.page * data.pageSize, data.totalCount)} of {data.totalCount} logs
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(data.page - 1)}
            disabled={data.page <= 1}
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(data.page + 1)}
            disabled={data.page >= data.totalPages}
          >
            Next
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
