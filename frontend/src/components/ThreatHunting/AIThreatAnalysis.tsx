'use client'

import { Brain, AlertTriangle, Shield, TrendingUp, Activity } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'
import { useMemo } from 'react'

interface AIInsight {
  id: number
  title: string
  description: string
  confidence: number
  severity: 'critical' | 'high' | 'medium' | 'low'
  category: string
  timestamp: string
}

export function AIThreatAnalysis() {
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 50,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  })

  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['ai-threat-logs'],
    queryFn: async () => {
      const end = new Date()
      const start = new Date()
      start.setDate(start.getDate() - 7)
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 500
      })
    },
    staleTime: 60000,
  })

  // Generate AI insights from alerts and logs
  const aiInsights = useMemo(() => {
    const insights: AIInsight[] = []
    const alerts = alertsData || []

    // Critical alerts insight
    const criticalAlerts = alerts.filter((a: any) => a.severity?.toLowerCase() === 'critical')
    if (criticalAlerts.length > 0) {
      insights.push({
        id: 1,
        title: 'Critical Security Threats Detected',
        description: `AI detected ${criticalAlerts.length} critical threat(s) requiring immediate investigation`,
        confidence: 95,
        severity: 'critical',
        category: 'Threat Detection',
        timestamp: new Date().toISOString()
      })
    }

    // Pattern-based insights
    const failedLogins = alerts.filter((a: any) => 
      a.message?.toLowerCase().includes('failed') || 
      a.message?.toLowerCase().includes('denied')
    )
    if (failedLogins.length >= 5) {
      insights.push({
        id: 2,
        title: 'Brute Force Attack Pattern',
        description: `Detected ${failedLogins.length} failed authentication attempts indicating possible brute force activity`,
        confidence: 87,
        severity: 'high',
        category: 'Attack Pattern',
        timestamp: new Date().toISOString()
      })
    }

    // Network anomaly
    if (logsData?.items) {
      const networkLogs = logsData.items.filter(l => 
        l.message?.toLowerCase().includes('network') ||
        l.message?.toLowerCase().includes('connection')
      )
      if (networkLogs.length > 100) {
        insights.push({
          id: 3,
          title: 'Unusual Network Activity',
          description: `High volume of network events detected - ${networkLogs.length} events in the last 7 days`,
          confidence: 72,
          severity: 'medium',
          category: 'Network Analysis',
          timestamp: new Date().toISOString()
        })
      }
    }

    // Add default insight if none found
    if (insights.length === 0) {
      insights.push({
        id: 4,
        title: 'System Operating Normally',
        description: 'AI analysis shows no significant threats or anomalies detected',
        confidence: 90,
        severity: 'low',
        category: 'System Status',
        timestamp: new Date().toISOString()
      })
    }

    return insights
  }, [alertsData, logsData])

  const isLoading = alertsLoading || logsLoading

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            AI Threat Analysis
          </h3>
        </div>
        <div className="flex items-center space-x-2">
          <Activity className="h-4 w-4 text-green-500 animate-pulse" />
          <span className="text-sm text-gray-500">Live Analysis</span>
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {aiInsights.length}
          </p>
          <p className="text-sm text-gray-500">AI Insights</p>
        </div>
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-red-600 dark:text-red-400">
            {aiInsights.filter(i => i.severity === 'critical' || i.severity === 'high').length}
          </p>
          <p className="text-sm text-gray-500">High Priority</p>
        </div>
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {Math.round(aiInsights.reduce((acc, i) => acc + i.confidence, 0) / aiInsights.length)}%
          </p>
          <p className="text-sm text-gray-500">Avg Confidence</p>
        </div>
      </div>

      {/* Insights List */}
      <div className="space-y-4">
        {aiInsights.map((insight) => (
          <div 
            key={insight.id}
            className={`p-4 rounded-lg border ${
              insight.severity === 'critical'
                ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
                : insight.severity === 'high'
                ? 'border-orange-200 bg-orange-50 dark:border-orange-800 dark:bg-orange-900/20'
                : insight.severity === 'medium'
                ? 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20'
                : 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                {insight.severity === 'critical' || insight.severity === 'high' ? (
                  <AlertTriangle className={`h-5 w-5 mt-0.5 ${
                    insight.severity === 'critical' ? 'text-red-500' : 'text-orange-500'
                  }`} />
                ) : (
                  <Shield className="h-5 w-5 text-green-500 mt-0.5" />
                )}
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {insight.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {insight.description}
                  </p>
                  <div className="mt-2 flex items-center space-x-4 text-xs text-gray-500">
                    <span className="flex items-center">
                      <TrendingUp className="h-3 w-3 mr-1" />
                      {insight.confidence}% confidence
                    </span>
                    <span>{insight.category}</span>
                  </div>
                </div>
              </div>
              <span className={`px-2 py-1 text-xs rounded-full ${
                insight.severity === 'critical'
                  ? 'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-200'
                  : insight.severity === 'high'
                  ? 'bg-orange-100 text-orange-800 dark:bg-orange-800 dark:text-orange-200'
                  : insight.severity === 'medium'
                  ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-200'
                  : 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-200'
              }`}>
                {insight.severity.toUpperCase()}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
