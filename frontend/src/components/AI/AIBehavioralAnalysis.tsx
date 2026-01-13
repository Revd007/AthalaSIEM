'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useBehavioralAnalytics } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'

export function AIBehavioralAnalysis() {
  const { data, isLoading } = useBehavioralAnalytics()

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    )
  }

  const behaviorData = data?.behaviorData || []
  const anomalies = data?.anomalies || []

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Behavioral Analysis
      </h3>
      
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={behaviorData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Line 
              type="monotone" 
              dataKey="normalScore" 
              stroke="#10b981" 
              name="Baseline"
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="userScore" 
              stroke="#ef4444" 
              name="Actual"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">
        Detected Anomalies
      </h4>
      
      <div className="space-y-3">
        {anomalies.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-sm">
            No anomalies detected
          </p>
        ) : (
          anomalies.map((anomaly) => (
            <div 
              key={anomaly.id}
              className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
            >
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {anomaly.user}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {anomaly.activity}
                </p>
              </div>
              <div className="text-right">
                <span className={`text-sm font-medium ${
                  anomaly.riskScore >= 80 ? 'text-red-500' :
                  anomaly.riskScore >= 60 ? 'text-yellow-500' :
                  'text-green-500'
                }`}>
                  {anomaly.riskScore}%
                </span>
                <p className="text-xs text-gray-400">{anomaly.time}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
