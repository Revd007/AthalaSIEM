'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useEventsOverTime } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

export function EventsTimeline() {
  const { data: timeline, isLoading } = useEventsOverTime(24)

  const chartData = (timeline ?? []).map(p => ({
    time: p.time,
    events: p.total,
    anomalies: p.errors,
  }))

  const peakEvents = chartData.length > 0 ? Math.max(...chartData.map(d => d.events)) : 0
  const peakIdx = chartData.findIndex(d => d.events === peakEvents)
  const peakTime = peakIdx >= 0 ? chartData[peakIdx]?.time ?? 'N/A' : 'N/A'
  const totalAnomalies = chartData.reduce((s, d) => s + d.anomalies, 0)

  return (
    <DashboardCard title="Events Timeline" icon={Clock}>
      <div className="h-[300px]">
        {isLoading ? (
          <Skeleton className="h-full w-full" />
        ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" tick={{ fontSize: 12 }} tickLine={false} />
            <YAxis tick={{ fontSize: 12 }} tickLine={false} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: 'none',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
              }}
            />
            <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={false} name="Events" />
            <Line type="monotone" dataKey="anomalies" stroke="#ef4444" strokeWidth={2} dot={false} name="Anomalies" />
          </LineChart>
        </ResponsiveContainer>
        )}
      </div>

      {/* Timeline Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-blue-500 mr-2" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Normal Events</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-2" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Anomalies</span>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="text-sm text-gray-500 dark:text-gray-400">Peak Events</div>
          <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">{peakEvents}</div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">at {peakTime}</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="text-sm text-gray-500 dark:text-gray-400">Total Anomalies</div>
          <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">{totalAnomalies}</div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">in last 24 hours</div>
        </div>
      </div>
    </DashboardCard>
  )
}
