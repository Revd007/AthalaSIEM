'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { LineChart as LineChartIcon, Brain, AlertTriangle, Target, Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { StatsCard } from '@/components/ui/StatsCard'

interface Prediction {
  id: string
  type: string
  probability: number
  impact: 'critical' | 'high' | 'medium' | 'low'
  timeframe: string
  details: string
  indicators: string[]
  mitigation: string[]
}

const mockTimeSeriesData = Array.from({ length: 24 }, (_, i) => ({
  time: `${i}:00`,
  predictedThreats: Math.floor(Math.random() * 50),
  confidence: 75 + Math.random() * 20
}))

const mockPredictions: Prediction[] = [
  {
    id: '1',
    type: 'Data Breach',
    probability: 78.5,
    impact: 'critical',
    timeframe: '24-48 hours',
    details: 'Potential data exfiltration based on unusual access patterns',
    indicators: [
      'Increased failed login attempts',
      'Unusual data transfer volumes',
      'Access from new locations'
    ],
    mitigation: [
      'Enable additional authentication factors',
      'Review access permissions',
      'Monitor sensitive data access'
    ]
  },
  {
    id: '2',
    type: 'DDoS Attack',
    probability: 65.2,
    impact: 'high',
    timeframe: '12-24 hours',
    details: 'Possible volumetric attack based on traffic analysis',
    indicators: [
      'Increased bandwidth usage',
      'Pattern matching previous attacks',
      'Suspicious source IPs'
    ],
    mitigation: [
      'Scale infrastructure',
      'Update firewall rules',
      'Enable DDoS protection'
    ]
  }
]

export function PredictiveAnalysis() {
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null)

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Active Predictions"
          value="8"
          change="+2"
          trend="up"
          icon={Brain}
          color="blue"
        />
        <StatsCard
          title="Prediction Accuracy"
          value="92.4%"
          change="+1.2%"
          trend="up"
          icon={Target}
          color="green"
        />
        <StatsCard
          title="Time to Detection"
          value="1.8h"
          change="-0.3h"
          trend="down"
          icon={Clock}
          color="yellow"
        />
        <StatsCard
          title="Critical Threats"
          value="3"
          change="+1"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
      </div>

      {/* Prediction Timeline */}
      <DashboardCard title="Prediction Timeline" icon={LineChartIcon}>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockTimeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time"
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                yAxisId="left"
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                tick={{ fontSize: 12 }}
              />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          Time: {payload[0].payload.time}
                        </p>
                        <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                          Threats: {payload[0].value}
                        </p>
                        <p className="text-sm font-medium text-green-600 dark:text-green-400">
                          Confidence: {typeof payload[1]?.value === 'number' ? payload[1].value.toFixed(1) : 'N/A'}%
                        </p>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="predictedThreats"
                stroke="#3b82f6"
                name="Predicted Threats"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="confidence"
                stroke="#10b981"
                name="Confidence"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </DashboardCard>

      {/* Active Predictions */}
      <DashboardCard title="Active Predictions" icon={AlertTriangle}>
        <div className="space-y-4">
          {mockPredictions.map((prediction) => (
            <div
              key={prediction.id}
              className={`p-4 rounded-lg border cursor-pointer transition-colors
                ${selectedPrediction?.id === prediction.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              onClick={() => setSelectedPrediction(prediction)}
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">
                    {prediction.type}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {prediction.details}
                  </p>
                </div>
                <span
                  className={`px-2 py-1 text-xs rounded-full ${
                    prediction.impact === 'critical'
                      ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                      : 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200'
                  }`}
                >
                  {prediction.probability}% probability
                </span>
              </div>

              {selectedPrediction?.id === prediction.id && (
                <div className="mt-4 space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Key Indicators
                    </h4>
                    <ul className="list-disc list-inside space-y-1">
                      {prediction.indicators.map((indicator, index) => (
                        <li
                          key={index}
                          className="text-sm text-gray-600 dark:text-gray-300"
                        >
                          {indicator}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Recommended Actions
                    </h4>
                    <ul className="list-disc list-inside space-y-1">
                      {prediction.mitigation.map((action, index) => (
                        <li
                          key={index}
                          className="text-sm text-gray-600 dark:text-gray-300"
                        >
                          {action}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </DashboardCard>
    </div>
  )
} 