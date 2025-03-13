'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Brain, Lightbulb, AlertTriangle, TrendingUp } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface Insight {
  id: string
  title: string
  description: string
  category: 'security' | 'performance' | 'compliance' | 'threat'
  severity: 'critical' | 'high' | 'medium' | 'low'
  timestamp: string
  confidence: number
  recommendations: string[]
}

const mockInsights: Insight[] = [
  {
    id: '1',
    title: 'Potential Data Exfiltration Pattern',
    description: 'Unusual data transfer patterns detected from multiple endpoints',
    category: 'security',
    severity: 'high',
    timestamp: new Date().toISOString(),
    confidence: 89,
    recommendations: [
      'Review network traffic patterns',
      'Investigate affected endpoints',
      'Update DLP policies'
    ]
  },
  {
    id: '2',
    title: 'Authentication Anomaly Cluster',
    description: 'Multiple failed login attempts across different systems',
    category: 'threat',
    severity: 'critical',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    confidence: 94,
    recommendations: [
      'Enable additional authentication factors',
      'Review access logs',
      'Update security policies'
    ]
  }
]

const mockTrendData = Array.from({ length: 7 }, (_, i) => ({
  date: new Date(Date.now() - i * 86400000).toLocaleDateString(),
  insights: Math.floor(Math.random() * 50),
  accuracy: 75 + Math.random() * 20
})).reverse()

export function AIInsights() {
  const [selectedInsight, setSelectedInsight] = useState<Insight | null>(null)

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Insights"
          value="156"
          change="+12"
          trend="up"
          icon={Brain}
          color="blue"
        />
        <StatsCard
          title="Critical Findings"
          value="8"
          change="+2"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="New Patterns"
          value="23"
          change="+5"
          trend="up"
          icon={Lightbulb}
          color="yellow"
        />
        <StatsCard
          title="Accuracy Rate"
          value="97.2%"
          change="+0.8%"
          trend="up"
          icon={TrendingUp}
          color="green"
        />
      </div>

      {/* Insights Trend */}
      <DashboardCard title="Insights Trend" icon={Brain}>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockTrendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="insights"
                stroke="#3b82f6"
                name="Insights"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="accuracy"
                stroke="#10b981"
                name="Accuracy %"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </DashboardCard>

      {/* Latest Insights */}
      <DashboardCard title="Latest Insights" icon={Lightbulb}>
        <div className="space-y-4">
          {mockInsights.map((insight) => (
            <div
              key={insight.id}
              className={`p-4 rounded-lg border cursor-pointer ${
                selectedInsight?.id === insight.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
              }`}
              onClick={() => setSelectedInsight(insight)}
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white">
                    {insight.title}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {insight.description}
                  </p>
                </div>
                <span
                  className={`px-2 py-1 text-xs rounded-full ${
                    insight.severity === 'critical'
                      ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                      : insight.severity === 'high'
                      ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200'
                      : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                  }`}
                >
                  {insight.severity}
                </span>
              </div>

              {selectedInsight?.id === insight.id && (
                <div className="mt-4 space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                      Recommendations
                    </h4>
                    <ul className="list-disc list-inside space-y-1">
                      {insight.recommendations.map((rec, index) => (
                        <li
                          key={index}
                          className="text-sm text-gray-600 dark:text-gray-300"
                        >
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500 dark:text-gray-400">
                      Confidence Score
                    </span>
                    <span className="text-gray-900 dark:text-white font-medium">
                      {insight.confidence}%
                    </span>
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