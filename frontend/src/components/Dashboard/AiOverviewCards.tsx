'use client'

import { Shield, Activity, Target, Clock, Percent, Brain } from 'lucide-react'
import { useAiOverview } from '@/hooks/useAiData'
import { Skeleton } from '@/components/ui/skeleton'

export function AiOverviewCards() {
  const { data, isLoading, isError } = useAiOverview()

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <Skeleton key={i} className="h-28 rounded-lg" />
        ))}
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/30 p-4 text-red-700 dark:text-red-300">
        Failed to load AI overview. Check backend connection.
      </div>
    )
  }

  const cards = [
    { label: 'Active Threats', value: data.activeThreats.toLocaleString(), icon: Shield, color: 'red' },
    { label: 'Avg Confidence', value: `${data.avgConfidence}%`, icon: Percent, color: 'blue' },
    { label: 'Detection Rate (24h)', value: data.detectionRate24h.toLocaleString(), icon: Activity, color: 'green' },
    { label: 'Response Time', value: data.responseTime, icon: Clock, color: 'indigo' },
    { label: 'MITRE Coverage', value: `${data.mitreCoveragePercent}%`, icon: Target, color: 'purple' },
    { label: 'Latest Insights', value: (data.latestInsights?.length ?? 0).toString(), icon: Brain, color: 'amber' },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {cards.map(({ label, value, icon: Icon }) => (
        <div
          key={label}
          className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 shadow-sm"
        >
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{label}</p>
            <Icon className="h-5 w-5 text-gray-400" />
          </div>
          <p className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">{value}</p>
        </div>
      ))}
    </div>
  )
}
