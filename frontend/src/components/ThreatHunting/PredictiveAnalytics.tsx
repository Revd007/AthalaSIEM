'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { usePredictiveAnalytics } from '@/services/analytics-service'
import { Skeleton } from '@/components/ui/skeleton'
import { AlertTriangle, TrendingUp, Shield, Activity } from 'lucide-react'

export function PredictiveAnalytics() {
  const { data, isLoading } = usePredictiveAnalytics()

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full" />
        <div className="grid grid-cols-2 gap-4">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      </div>
    )
  }

  const predictions = data?.predictions || []
  const riskFactors = data?.riskFactors || []

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Predictive Analytics
        </h3>
        <div className="flex items-center space-x-2">
          <Activity className="h-5 w-5 text-blue-500" />
          <span className="text-sm text-gray-500">Real-time Analysis</span>
        </div>
      </div>
      
      {/* Prediction Chart */}
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="actual" 
              stroke="#10b981" 
              name="Actual Events"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="predicted" 
              stroke="#3b82f6" 
              name="Predicted Events"
              strokeWidth={2}
              strokeDasharray="5 5"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Risk Factors */}
      <h4 className="text-md font-medium text-gray-900 dark:text-white mb-4">
        Risk Factors
      </h4>
      
      <div className="space-y-4">
        {riskFactors.length === 0 ? (
          <div className="text-center text-gray-500 py-4">
            <Shield className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <p>No significant risk factors detected</p>
          </div>
        ) : (
          riskFactors.map((factor, index) => (
            <div 
              key={index}
              className={`p-4 rounded-lg border ${
                factor.impact === 'high' 
                  ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
                  : factor.impact === 'medium'
                  ? 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20'
                  : 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  {factor.impact === 'high' ? (
                    <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
                  ) : (
                    <TrendingUp className="h-5 w-5 text-yellow-500 mt-0.5" />
                  )}
                  <div>
                    <h5 className="font-medium text-gray-900 dark:text-white">
                      {factor.title}
                    </h5>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {factor.description}
                    </p>
                  </div>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  factor.impact === 'high'
                    ? 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-200'
                    : factor.impact === 'medium'
                    ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-200'
                    : 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-200'
                }`}>
                  {factor.impact.toUpperCase()}
                </span>
              </div>
              <div className="mt-3 pl-8">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  <span className="font-medium">Recommendation:</span> {factor.recommendation}
                </p>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Summary Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {predictions.reduce((acc, p) => acc + p.predicted, 0)}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Predicted Events</p>
        </div>
        <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {riskFactors.filter(f => f.impact === 'high').length}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">High Risk Factors</p>
        </div>
        <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            96%
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Prediction Accuracy</p>
        </div>
      </div>
    </div>
  )
}
