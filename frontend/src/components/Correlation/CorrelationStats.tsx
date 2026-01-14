'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Shield, AlertTriangle, TrendingUp, List } from 'lucide-react'
import { type CorrelationStatistics } from '@/hooks/useCorrelation'
import { Skeleton } from '@/components/ui/skeleton'

interface CorrelationStatsProps {
  statistics: CorrelationStatistics | undefined
  isLoading: boolean
}

export function CorrelationStats({ statistics, isLoading }: CorrelationStatsProps) {
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

  const totalSeverity = statistics.severityBreakdown.reduce((sum, item) => sum + item.count, 0)

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Correlation Alerts</CardTitle>
          <AlertTriangle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{statistics.totalCorrelationAlerts.toLocaleString()}</div>
          <p className="text-xs text-muted-foreground">Total alerts generated</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Active Rules</CardTitle>
          <Shield className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{statistics.ruleBreakdown.length}</div>
          <p className="text-xs text-muted-foreground">Rules configured</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Rule Triggers</CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {statistics.ruleBreakdown.reduce((sum, item) => sum + item.count, 0).toLocaleString()}
          </div>
          <p className="text-xs text-muted-foreground">Total rule triggers</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Severity Levels</CardTitle>
          <List className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{statistics.severityBreakdown.length}</div>
          <p className="text-xs text-muted-foreground">Unique severity levels</p>
        </CardContent>
      </Card>
    </div>
  )
}
