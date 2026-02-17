'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Activity, AlertTriangle, Shield, Clock } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { StatsCard } from './StatsCard'
import {
  useDashboardSummary,
  useEventsOverTime,
  useSeverityDistribution,
} from '@/services/analytics-service'
import { useAlerts } from '@/services/alert-service'
import { Skeleton } from '@/components/ui/skeleton'

interface SecurityEventsOverviewProps {
  timeRange: string
  filters: any
}

export function SecurityEventsOverview({ timeRange, filters }: SecurityEventsOverviewProps) {
  const hoursMap: Record<string, number> = { '1h': 1, '24h': 24, '7d': 168, '30d': 720 }
  const hours = hoursMap[timeRange] ?? 24

  // All data comes from pre-aggregated backend endpoints — no 10K-log client fetch
  const { data: summary, isLoading: summaryLoading } = useDashboardSummary()
  const { data: timeline, isLoading: timelineLoading } = useEventsOverTime(hours)
  const { data: severityDist } = useSeverityDistribution()
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({ limit: 100 })

  const isLoading = summaryLoading || timelineLoading

  // Stats derived from the lightweight summary endpoint
  const totalEvents  = summary?.totalLogs24h ?? 0
  const criticalEvents = summary?.criticalCount ?? 0
  const eps = summary?.eventsPerSecond ?? 0
  const resolvedAlerts = alertsData?.filter(a => a.status === 'Resolved').length ?? 0

  // Timeline data is already pre-bucketed by the backend
  const chartData = timeline ?? []

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Events (24h)"
          value={isLoading ? '...' : totalEvents > 1000 ? `${(totalEvents / 1000).toFixed(1)}K` : totalEvents.toLocaleString()}
          change={`${eps}/s`}
          icon={Activity}
          trend="up"
        />
        <StatsCard
          title="Critical / Errors"
          value={isLoading ? '...' : criticalEvents.toLocaleString()}
          change={totalEvents > 0 ? `${((criticalEvents / totalEvents) * 100).toFixed(1)}%` : '0%'}
          icon={AlertTriangle}
          trend={criticalEvents > 10 ? 'up' : 'down'}
          color="red"
        />
        <StatsCard
          title="Events/Second"
          value={isLoading ? '...' : `${eps}`}
          change="real-time"
          icon={Clock}
          trend="down"
          color="green"
        />
        <StatsCard
          title="Alerts Resolved"
          value={alertsLoading ? '...' : resolvedAlerts.toLocaleString()}
          change={`${alertsData?.length ?? 0} total`}
          icon={Shield}
          trend="up"
          color="blue"
        />
      </div>

      {/* Events Timeline Chart — plots total + errors + warnings from the backend */}
      <DashboardCard title="Events Over Time" icon={Activity}>
        <div className="h-[300px]">
          {isLoading ? (
            <Skeleton className="h-full w-full" />
          ) : chartData.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-400">
              No event data available for the selected range
            </div>
          ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Area
                type="monotone"
                dataKey="total"
                stackId="1"
                stroke="#3b82f6"
                fill="#dbeafe"
                name="Total Events"
              />
              <Area
                type="monotone"
                dataKey="errors"
                stackId="2"
                stroke="#ef4444"
                fill="#fee2e2"
                name="Errors/Critical"
              />
              <Area
                type="monotone"
                dataKey="warnings"
                stackId="2"
                stroke="#f59e0b"
                fill="#fef3c7"
                name="Warnings"
              />
            </AreaChart>
          </ResponsiveContainer>
          )}
        </div>
      </DashboardCard>
    </div>
  )
}
