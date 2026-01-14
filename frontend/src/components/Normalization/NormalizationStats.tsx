'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Database, CheckCircle2, TrendingUp, AlertCircle } from 'lucide-react'
import { type NormalizationStatistics } from '@/hooks/useNormalization'
import { Skeleton } from '@/components/ui/skeleton'

interface NormalizationStatsProps {
  statistics: NormalizationStatistics | undefined
  isLoading: boolean
}

export function NormalizationStats({ statistics, isLoading }: NormalizationStatsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-4 w-4" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-8 w-32 mb-2" />
              <Skeleton className="h-4 w-48" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  if (!statistics) {
    return null
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Logs</CardTitle>
          <Database className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{statistics.totalLogs.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground">All log entries</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Normalized Logs</CardTitle>
          <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{statistics.normalizedLogs.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground">With ECS fields</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Normalization Rate</CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{statistics.normalizationRate.toFixed(1)}%</div>
          <p className="text-xs text-muted-foreground">
            {statistics.normalizedLogs} of {statistics.totalLogs}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Event Types</CardTitle>
          <AlertCircle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{statistics.eventTypeDistribution.length}</div>
          <p className="text-xs text-muted-foreground">Unique categories</p>
        </CardContent>
      </Card>
    </div>
  )
}
