import React from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../../ui/table'
import { Skeleton } from '../../ui/skeleton'
import { Alert } from '../../../types/alert'

interface AlertTableProps {
  alerts?: Alert[]
  isLoading: boolean
}

export function AlertTable({ alerts, isLoading }: AlertTableProps) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    )
  }

  if (!alerts?.length) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No alerts found
      </div>
    )
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Severity</TableHead>
          <TableHead>Title</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Source</TableHead>
          <TableHead>Timestamp</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {alerts.map((alert) => (
          <TableRow key={alert.id}>
            <TableCell>{alert.severity}</TableCell>
            <TableCell>{alert.title}</TableCell>
            <TableCell>{alert.status}</TableCell>
            <TableCell>{alert.source}</TableCell>
            <TableCell>{new Date(alert.timestamp).toLocaleString()}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
} 