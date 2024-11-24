import React, { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle } from '../ui/card'
import { axiosInstance } from '../../lib/axios'

interface AIInsight {
  type: string
  description: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  confidence: number
  timestamp: string
}

export function AIInsights() {
  const [insights, setInsights] = useState<AIInsight[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const response = await axiosInstance.get('/api/v1/ai/insights')
        setInsights(response.data)
      } catch (error) {
        console.error('Failed to fetch AI insights:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchInsights()
    const interval = setInterval(fetchInsights, 300000) // Update every 5 minutes
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div>Loading AI insights...</div>

  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Insights</CardTitle>
      </CardHeader>
      <div className="p-6">
        {insights.map((insight, index) => (
          <div
            key={index}
            className={`mb-4 p-4 rounded-lg border ${
              insight.severity === 'critical' ? 'border-red-500 bg-red-50' :
              insight.severity === 'high' ? 'border-orange-500 bg-orange-50' :
              insight.severity === 'medium' ? 'border-yellow-500 bg-yellow-50' :
              'border-blue-500 bg-blue-50'
            }`}
          >
            <div className="flex justify-between items-start">
              <div>
                <h4 className="font-medium">{insight.type}</h4>
                <p className="text-sm text-gray-600 mt-1">{insight.description}</p>
              </div>
              <span className="text-sm text-gray-500">
                {new Date(insight.timestamp).toLocaleString()}
              </span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <span className="text-sm text-gray-500">
                Confidence: {insight.confidence}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}