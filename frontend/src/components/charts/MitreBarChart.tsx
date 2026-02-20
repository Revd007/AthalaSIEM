'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export interface MitreBarItem {
  technique: string
  count: number
}

interface MitreBarChartProps {
  data: MitreBarItem[]
  height?: number
}

export function MitreBarChart({ data, height = 280 }: MitreBarChartProps) {
  const chartData = data.map((d) => ({
    name: d.technique.length > 20 ? d.technique.slice(0, 20) + '…' : d.technique,
    fullName: d.technique,
    count: d.count,
  }))

  if (chartData.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-500 dark:text-gray-400"
        style={{ height }}
      >
        No MITRE technique data
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
        <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={78} tick={{ fontSize: 10 }} />
        <Tooltip
          formatter={(value: number) => [value, 'Count']}
          labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName ?? ''}
        />
        <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
