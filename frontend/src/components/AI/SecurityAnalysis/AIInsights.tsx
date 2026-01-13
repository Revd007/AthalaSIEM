'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Brain, Lightbulb, AlertTriangle, TrendingUp } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'

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

export function AIInsights() {
  const [selectedInsight, setSelectedInsight] = useState<Insight | null>(null)

  // Fetch alerts for insights
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 50,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for pattern analysis
  const { data: logsData } = useQuery({
    queryKey: ['ai-insights-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 7);
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 1000
      });
    },
    refetchInterval: 60000,
  });

  // Generate insights from alerts
  const insights = useMemo(() => {
    if (!alertsData?.items) return [];

    const insights: Insight[] = [];
    const alerts = alertsData.items;

    // Critical alerts insight
    const criticalAlerts = alerts.filter(a => a.severity?.toLowerCase() === 'critical');
    if (criticalAlerts.length > 0) {
      insights.push({
        id: '1',
        title: 'Critical Security Alerts Detected',
        description: `${criticalAlerts.length} critical alert(s) require immediate attention`,
        category: 'threat',
        severity: 'critical',
        timestamp: criticalAlerts[0]?.timestamp || new Date().toISOString(),
        confidence: 95,
        recommendations: [
          'Review all critical alerts immediately',
          'Verify alert sources and agents',
          'Check for system compromises',
          'Engage incident response team if needed'
        ]
      });
    }

    // Failed login pattern
    const failedLogins = alerts.filter(a => 
      a.message?.toLowerCase().includes('failed login') ||
      a.message?.toLowerCase().includes('authentication failed')
    );
    if (failedLogins.length >= 3) {
      insights.push({
        id: '2',
        title: 'Authentication Anomaly Pattern',
        description: `Multiple failed login attempts detected (${failedLogins.length} events)`,
        category: 'security',
        severity: 'high',
        timestamp: failedLogins[0]?.timestamp || new Date().toISOString(),
        confidence: 85,
        recommendations: [
          'Review authentication logs',
          'Check for brute force attempts',
          'Enable account lockout policies',
          'Consider implementing MFA'
        ]
      });
    }

    // Network activity insight
    if (logsData?.items) {
      const networkLogs = logsData.items.filter(l => 
        l.message?.toLowerCase().includes('network') ||
        l.message?.toLowerCase().includes('connection')
      );
      if (networkLogs.length > 200) {
        insights.push({
          id: '3',
          title: 'Unusual Network Activity',
          description: `High volume of network-related events detected (${networkLogs.length} events)`,
          category: 'security',
          severity: 'medium',
          timestamp: new Date().toISOString(),
          confidence: 75,
          recommendations: [
            'Review network traffic patterns',
            'Check firewall rules',
            'Monitor for DDoS indicators',
            'Verify network access policies'
          ]
        });
      }
    }

    return insights;
  }, [alertsData, logsData]);

  // Generate trend data from alerts
  const trendData = useMemo(() => {
    if (!alertsData?.items) {
      return Array.from({ length: 7 }, (_, i) => ({
        date: new Date(Date.now() - i * 86400000).toLocaleDateString(),
        insights: 0,
        accuracy: 0
      })).reverse();
    }

    const dailyData: Record<string, number> = {};
    alertsData.items.forEach(alert => {
      if (alert.timestamp) {
        const date = new Date(alert.timestamp).toLocaleDateString();
        dailyData[date] = (dailyData[date] || 0) + 1;
      }
    });

    return Array.from({ length: 7 }, (_, i) => {
      const date = new Date(Date.now() - i * 86400000).toLocaleDateString();
      return {
        date,
        insights: dailyData[date] || 0,
        accuracy: dailyData[date] ? Math.min(95, 70 + dailyData[date] * 2) : 0
      };
    }).reverse();
  }, [alertsData]);

  const totalInsights = insights.length;
  const criticalFindings = insights.filter(i => i.severity === 'critical').length;

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Insights"
          value={totalInsights.toString()}
          change="+0"
          trend="up"
          icon={Brain}
          color="blue"
        />
        <StatsCard
          title="Critical Findings"
          value={criticalFindings.toString()}
          change="+0"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Total Alerts"
          value={(alertsData?.totalCount || 0).toString()}
          change="+0"
          trend="up"
          icon={Lightbulb}
          color="yellow"
        />
        <StatsCard
          title="High Confidence"
          value={insights.filter(i => i.confidence >= 85).length.toString()}
          change="+0"
          trend="up"
          icon={TrendingUp}
          color="green"
        />
      </div>

      {/* Insights Trend */}
      <DashboardCard title="Insights Trend" icon={Brain}>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trendData}>
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
          {alertsLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : insights.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              No insights available. System is operating normally.
            </div>
          ) : (
            insights.map((insight) => (
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
          )))}
        </div>
      </DashboardCard>
    </div>
  )
} 