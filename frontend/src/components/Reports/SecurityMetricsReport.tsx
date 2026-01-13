'use client'

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { useSecurityMetrics } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

export function SecurityMetricsReport() {
  const { data, isLoading } = useSecurityMetrics()

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  const securityMetrics = data?.monthlyData || []
  const kpis = data?.kpis || []

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Security Metrics Report
      </h3>
      
      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {kpis.map((kpi, index) => (
          <div 
            key={index}
            className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg"
          >
            <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              {kpi.title}
            </p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
              {kpi.value}
            </p>
            <span className={`text-sm ${
              kpi.trend === 'up' ? 'text-green-500' : 'text-red-500'
            }`}>
              {kpi.change}
            </span>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={securityMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="incidents" fill="#ef4444" name="Incidents" />
            <Bar dataKey="resolved" fill="#10b981" name="Resolved" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* MTTR Trend */}
      <div className="mt-6">
        <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">
          Mean Time to Resolve (Hours)
        </h4>
        <div className="flex items-center space-x-4">
          {securityMetrics.slice(-6).map((metric, index) => (
            <div key={index} className="text-center">
              <p className="text-lg font-bold text-blue-500">{metric.mttr}h</p>
              <p className="text-xs text-gray-500">{metric.month}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
