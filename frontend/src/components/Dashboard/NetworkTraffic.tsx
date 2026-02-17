'use client'

import React, { useMemo } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { useEventsOverTime } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

export function NetworkTraffic() {
  const { data: timeline, isLoading } = useEventsOverTime(24)

  const chartData = useMemo(() => {
    if (!timeline || timeline.length === 0) return []
    return timeline.map(p => ({
      time: p.time,
      'Normal Events': p.normal,
      'Anomalies': p.errors + p.warnings,
    }))
  }, [timeline])

  const totalEvents = useMemo(() => {
    if (!timeline) return 0
    return timeline.reduce((s, p) => s + p.total, 0)
  }, [timeline])

  const totalAnomalies = useMemo(() => {
    if (!timeline) return 0
    return timeline.reduce((s, p) => s + p.errors + p.warnings, 0)
  }, [timeline])

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Events Timeline (24h)</h2>
        <Skeleton className="h-80 w-full" />
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Events Timeline (24h)</h2>
      {chartData.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-gray-400">
          No event data — logs will appear as they are ingested
        </div>
      ) : (
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" tick={{ fontSize: 11 }} interval={2} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="Normal Events" stackId="1" stroke="#3b82f6" fill="#93c5fd" />
            <Area type="monotone" dataKey="Anomalies" stackId="1" stroke="#ef4444" fill="#fca5a5" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      )}
      <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
          <div className="text-gray-500 dark:text-gray-400">Total Events</div>
          <div className="text-xl font-semibold text-gray-900 dark:text-white">{totalEvents.toLocaleString()}</div>
        </div>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
          <div className="text-gray-500 dark:text-gray-400">Total Anomalies</div>
          <div className="text-xl font-semibold text-gray-900 dark:text-white">{totalAnomalies.toLocaleString()}</div>
        </div>
      </div>
    </div>
  )
}
