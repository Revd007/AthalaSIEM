'use client'

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'
import { useDeviceAnalytics } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

const DEVICE_COLORS = ['#3b82f6', '#10b981', '#8b5cf6']

export function DeviceAnalytics() {
  const { data, isLoading } = useDeviceAnalytics()

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-2 gap-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    )
  }

  const deviceData = data?.deviceData || []
  const severityData = data?.severityData || []

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Device Analytics
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Device Distribution */}
        <div>
          <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-3">
            Device Distribution
          </h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={deviceData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {deviceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={DEVICE_COLORS[index % DEVICE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Severity Distribution */}
        <div>
          <h4 className="text-md font-medium text-gray-700 dark:text-gray-300 mb-3">
            Alert Severity
          </h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={severityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {deviceData.reduce((acc, curr) => acc + curr.value, 0)}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Total Devices</p>
        </div>
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-red-600 dark:text-red-400">
            {severityData.find(s => s.name === 'Critical')?.value || 0}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Critical Alerts</p>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {Math.round(((severityData.find(s => s.name === 'Low')?.value || 0) / 
              severityData.reduce((acc, curr) => acc + curr.value, 1)) * 100)}%
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Low Severity Rate</p>
        </div>
      </div>
    </div>
  )
}
