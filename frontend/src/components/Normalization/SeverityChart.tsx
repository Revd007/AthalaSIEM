'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useMemo } from 'react'

interface SeverityChartProps {
  data: Array<{ severity: number | null; count: number }>
}

const SEVERITY_LABELS: Record<number, string> = {
  1: 'Debug',
  2: 'Info',
  4: 'Warning',
  7: 'Error',
  10: 'Critical',
}

export function SeverityChart({ data }: SeverityChartProps) {
  const chartData = useMemo(() => {
    return data
      .map((item) => ({
        severity: item.severity ?? 0,
        label: SEVERITY_LABELS[item.severity ?? 0] || `Level ${item.severity ?? 0}`,
        count: item.count,
      }))
      .sort((a, b) => a.severity - b.severity)
  }, [data])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No data available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="label" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="count" fill="#8884d8" />
      </BarChart>
    </ResponsiveContainer>
  )
}
