'use client'

import { FileWarning, AlertTriangle, CheckCircle, Server, FolderOpen } from 'lucide-react'
import { useFIMStats } from '@/services/fim-service'
import { Skeleton } from '@/components/ui/skeleton'

interface FIMStatsCardsProps {
  agentId?: string
  days?: number
}

export function FIMStatsCards({ agentId, days = 7 }: FIMStatsCardsProps) {
  const { data, isLoading, isError } = useFIMStats(agentId, days)

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} className="h-28 rounded-lg" />
        ))}
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load FIM statistics. Check backend connection.
      </div>
    )
  }

  const criticalCount =
    data.eventsBySeverity?.find((s) => s.severity === 'Critical')?.count ?? 0
  const highCount =
    data.eventsBySeverity?.find((s) => s.severity === 'High')?.count ?? 0

  const cards = [
    {
      label: `Total events (${days}d)`,
      value: data.totalEvents.toLocaleString(),
      icon: FileWarning,
    },
    {
      label: 'Unacknowledged',
      value: data.unacknowledgedEvents.toLocaleString(),
      icon: AlertTriangle,
    },
    {
      label: 'Acknowledged',
      value: data.acknowledgedEvents.toLocaleString(),
      icon: CheckCircle,
    },
    {
      label: 'Critical / High',
      value: `${criticalCount} / ${highCount}`,
      icon: AlertTriangle,
    },
    {
      label: 'Agents with events',
      value: (data.eventsByAgent?.length ?? 0).toString(),
      icon: Server,
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {cards.map(({ label, value, icon: Icon }) => (
        <div
          key={label}
          className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 shadow-sm"
        >
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
              {label}
            </p>
            <Icon className="h-5 w-5 text-gray-400" />
          </div>
          <p className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">
            {value}
          </p>
        </div>
      ))}
    </div>
  )
}
