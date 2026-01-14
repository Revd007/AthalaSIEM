'use client'

import { useQuery } from '@tanstack/react-query'
import { api, endpoints } from '@/lib/api'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { AlertTriangle } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ExternalLink } from 'lucide-react'

interface Alert {
  id: string
  title: string
  description: string
  severity: string
  status: string
  createdAt: string
  agentId: string
  source: string
}

interface CorrelationAlertsProps {
  dateRange: { start?: string; end?: string }
}

export function CorrelationAlerts({ dateRange }: CorrelationAlertsProps) {
  const { data: alerts, isLoading } = useQuery<Alert[]>({
    queryKey: ['correlation', 'alerts', dateRange],
    queryFn: async () => {
      const params = new URLSearchParams()
      params.append('source', 'CorrelationEngine')
      if (dateRange.start) params.append('startDate', dateRange.start)
      if (dateRange.end) params.append('endDate', dateRange.end)
      
      const response = await api.get<Alert[]>(`${endpoints.alerts?.list || '/api/alerts'}?${params.toString()}`)
      return response.data || []
    },
    refetchInterval: 10000,
  })

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    )
  }

  if (!alerts || alerts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <AlertTriangle className="h-12 w-12 mb-4 opacity-50" />
        <p>No correlation alerts found</p>
      </div>
    )
  }

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
      case 'high':
        return 'bg-red-500'
      case 'medium':
        return 'bg-yellow-500'
      case 'low':
        return 'bg-blue-500'
      default:
        return 'bg-gray-500'
    }
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Title</TableHead>
            <TableHead>Severity</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Created</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {alerts.map((alert) => (
            <TableRow key={alert.id}>
              <TableCell>
                <div>
                  <div className="font-medium">{alert.title}</div>
                  <div className="text-sm text-muted-foreground">{alert.description}</div>
                </div>
              </TableCell>
              <TableCell>
                <Badge className={getSeverityColor(alert.severity)} variant="default">
                  {alert.severity}
                </Badge>
              </TableCell>
              <TableCell>
                <Badge variant="outline">{alert.status}</Badge>
              </TableCell>
              <TableCell className="text-sm">
                {new Date(alert.createdAt).toLocaleString()}
              </TableCell>
              <TableCell>
                <Link href={`/dashboard/alerts/${alert.id}`}>
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
  )
}
