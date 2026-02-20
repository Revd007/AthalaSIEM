'use client'

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export interface PredictionTimelinePoint {
  time: string
  count: number
}

interface PredictionTimelineProps {
  data: PredictionTimelinePoint[]
  height?: number
}

function formatTimeLabel(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

export function PredictionTimeline({ data, height = 280 }: PredictionTimelineProps) {
  const chartData = data.map((d) => ({
    ...d,
    label: formatTimeLabel(d.time),
  }))

  if (chartData.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-500 dark:text-gray-400"
        style={{ height }}
      >
        No prediction timeline data
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
        <XAxis dataKey="label" tick={{ fontSize: 11 }} />
        <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
        <Tooltip
          formatter={(value: number) => [value, 'Count']}
          labelFormatter={(_, payload) => payload?.[0]?.payload?.time ?? ''}
        />
        <Area
          type="monotone"
          dataKey="count"
          stroke="#3b82f6"
          fill="#3b82f6"
          fillOpacity={0.3}
          strokeWidth={2}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
