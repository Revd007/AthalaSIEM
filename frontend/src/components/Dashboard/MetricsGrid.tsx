'use client'

import React from 'react'
import { AlertCircle, Shield, Activity, Network, Users, Clock } from 'lucide-react'
import { useDashboardSummary } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

export function MetricsGrid() {
  const { data: summary, isLoading } = useDashboardSummary()

  const eps = summary?.eventsPerSecond ?? 0
  const criticalAlerts = summary?.criticalCount ?? 0
  const activeAgents = summary?.onlineAgents ?? 0
  const totalLogs1h = summary?.totalLogs1h ?? 0
  const totalLogs24h = summary?.totalLogs24h ?? 0
  const totalAlerts = summary?.totalAlerts ?? 0

  const metrics = [
    {
      label: 'Events/sec',
      value: eps.toFixed(1),
      icon: Activity,
      detail: `${totalLogs1h.toLocaleString()} in 1h`,
      color: 'blue'
    },
    {
      label: 'Critical / Errors',
      value: criticalAlerts.toLocaleString(),
      icon: AlertCircle,
      detail: `${totalAlerts} alerts`,
      color: 'red'
    },
    {
      label: 'Active Agents',
      value: `${activeAgents}/${summary?.totalAgents ?? 0}`,
      icon: Users,
      detail: 'online / total',
      color: 'purple'
    },
    {
      label: 'Total Logs (24h)',
      value: totalLogs24h > 1000 ? `${(totalLogs24h / 1000).toFixed(1)}K` : totalLogs24h.toLocaleString(),
      icon: Shield,
      detail: `${totalLogs1h.toLocaleString()} last hour`,
      color: 'green'
    },
    {
      label: 'Network Load',
      value: `${eps.toFixed(1)}/s`,
      icon: Network,
      detail: 'events per second',
      color: 'orange'
    },
    {
      label: 'Avg Response',
      value: '<1s',
      icon: Clock,
      detail: 'agent → backend',
      color: 'indigo'
    },
  ]

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[1, 2, 3, 4, 5, 6].map(i => (
          <Skeleton key={i} className="h-28 w-full rounded-lg" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {metrics.map((metric) => (
        <div key={metric.label} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">{metric.label}</p>
              <p className="text-2xl font-semibold mt-1 text-gray-900 dark:text-white">{metric.value}</p>
            </div>
            <div className={`rounded-full p-3 bg-${metric.color}-50 dark:bg-${metric.color}-900/20`}>
              <metric.icon className={`w-6 h-6 text-${metric.color}-500`} />
            </div>
          </div>
          <div className="flex items-center mt-4">
            <span className="text-sm text-gray-500 dark:text-gray-400">{metric.detail}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
