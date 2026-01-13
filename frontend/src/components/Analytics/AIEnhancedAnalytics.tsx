'use client'

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import { useAIAnalytics } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

export function AIEnhancedAnalytics() {
  const { data, isLoading } = useAIAnalytics()

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  const anomalyData = data?.anomalyData || []
  const threatDistribution = data?.threatDistribution || []

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        AI-Enhanced Analytics
      </h3>
      
      {/* Anomaly Detection Chart */}
      <div>
        <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-3">
          Anomaly Detection
        </h4>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={anomalyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="baseline" 
                stackId="1"
                stroke="#10b981" 
                fill="#10b981"
                fillOpacity={0.3}
                name="Baseline"
              />
              <Area 
                type="monotone" 
                dataKey="actual" 
                stackId="2"
                stroke="#ef4444" 
                fill="#ef4444"
                fillOpacity={0.3}
                name="Actual"
              />
              <Area 
                type="monotone" 
                dataKey="predicted" 
                stackId="3"
                stroke="#3b82f6" 
                fill="#3b82f6"
                fillOpacity={0.3}
                name="Predicted"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Threat Distribution */}
      <div>
        <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-3">
          Threat Distribution
        </h4>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={threatDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {threatDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* AI Insights Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {anomalyData.filter(d => d.actual > d.baseline * 1.2).length}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Anomalies Detected</p>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {Math.round(anomalyData.reduce((acc, d) => acc + Math.abs(d.predicted - d.actual), 0) / anomalyData.length) || 0}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Avg Prediction Error</p>
        </div>
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-red-600 dark:text-red-400">
            {threatDistribution.reduce((acc, curr) => acc + curr.value, 0)}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Total Threats</p>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            94%
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Model Accuracy</p>
        </div>
      </div>
    </div>
  )
}
