'use client'

import React from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'
import { useEventsDistribution, useSeverityDistribution } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

export function SecurityEvents() {
  const { data: categoryData, isLoading: catLoading, isError: catError } = useEventsDistribution()
  const { data: severityData, isLoading: sevLoading, isError: sevError } = useSeverityDistribution()

  const isLoading = catLoading || sevLoading
  const isError = catError || sevError

  // Prefer category distribution; fall back to severity if only one category exists
  const distributionData = React.useMemo(() => {
    if (categoryData && categoryData.length > 1) return categoryData
    if (severityData && severityData.length > 0) {
      return severityData.map(s => ({ name: s.name, value: s.value }))
    }
    return categoryData ?? []
  }, [categoryData, severityData])

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Security Events Distribution</h2>
        <Skeleton className="h-80 w-full" />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Security Events Distribution</h2>
        <div className="h-80 flex items-center justify-center text-gray-500 dark:text-gray-400">
          Failed to load distribution data
        </div>
      </div>
    )
  }

  if (distributionData.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Security Events Distribution</h2>
        <div className="h-80 flex items-center justify-center text-gray-500">
          No events data — logs will appear as they are ingested
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Security Events Distribution</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={distributionData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {distributionData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number) => [`${value} events`, 'Count']} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
