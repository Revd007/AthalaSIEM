'use client'

import { useMemo } from 'react';
import { Brain, Shield, AlertTriangle, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { useAlerts } from '@/services/alert-service';
import { logService } from '@/services/log-service';
import { Skeleton } from '@/components/ui/skeleton';

export function AISecurityFeatures() {
  // Fetch alerts for AI analysis
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 100,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for anomaly detection
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['ai-security-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setHours(start.getHours() - 24);
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 1000
      });
    },
    refetchInterval: 60000,
  });

  // Generate anomaly scores from logs
  const anomalyScores = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 6 }, (_, i) => ({
        time: `${i * 4}:00`,
        score: 0
      }));
    }

    const hourlyData: Record<number, { total: number; high: number }> = {};
    
    logsData.items.forEach(log => {
      if (log.timestamp) {
        const hour = new Date(log.timestamp).getHours();
        const bucket = Math.floor(hour / 4) * 4;
        if (!hourlyData[bucket]) {
          hourlyData[bucket] = { total: 0, high: 0 };
        }
        hourlyData[bucket].total++;
        if (log.severity === 'High' || log.severity === 'Critical') {
          hourlyData[bucket].high++;
        }
      }
    });

    return Array.from({ length: 6 }, (_, i) => {
      const bucket = i * 4;
      const data = hourlyData[bucket] || { total: 0, high: 0 };
      return {
        time: `${bucket.toString().padStart(2, '0')}:00`,
        score: data.total > 0 ? Math.min(1, data.high / data.total * 3) : 0
      };
    });
  }, [logsData]);

  // Generate AI insights from alerts
  const insights = useMemo(() => {
    if (!alertsData?.items) return [];

    const insights: Array<{
      id: number;
      type: string;
      title: string;
      description: string;
      severity: string;
      confidence: number;
    }> = [];

    const alerts = alertsData.items;

    // Check for authentication issues
    const authAlerts = alerts.filter(a => 
      a.message?.toLowerCase().includes('login') ||
      a.message?.toLowerCase().includes('auth')
    );
    if (authAlerts.length >= 3) {
      insights.push({
        id: 1,
        type: 'anomaly',
        title: 'Unusual Authentication Pattern',
        description: `Multiple authentication events detected (${authAlerts.length} events)`,
        severity: authAlerts.length >= 5 ? 'critical' : 'high',
        confidence: Math.min(0.95, 0.7 + authAlerts.length * 0.03)
      });
    }

    // Check for critical alerts
    const criticalAlerts = alerts.filter(a => 
      a.severity?.toLowerCase() === 'critical'
    );
    if (criticalAlerts.length > 0) {
      insights.push({
        id: 2,
        type: 'prediction',
        title: 'Critical Security Event',
        description: `${criticalAlerts.length} critical alert(s) detected requiring immediate attention`,
        severity: 'critical',
        confidence: 0.92
      });
    }

    // Check for network anomalies
    const networkAlerts = alerts.filter(a => 
      a.message?.toLowerCase().includes('network') ||
      a.message?.toLowerCase().includes('ddos') ||
      a.message?.toLowerCase().includes('traffic')
    );
    if (networkAlerts.length >= 2) {
      insights.push({
        id: 3,
        type: 'prediction',
        title: 'Network Anomaly Detected',
        description: 'Unusual network patterns suggest potential attack activity',
        severity: 'high',
        confidence: 0.85
      });
    }

    return insights;
  }, [alertsData]);

  const isLoading = alertsLoading || logsLoading;

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <div className="flex items-center space-x-2 mb-6">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">AI Security Analysis</h2>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">AI Security Analysis</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Anomaly Detection</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={anomalyScores}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 1]} />
                <Tooltip 
                  formatter={(value: number) => [`${(value * 100).toFixed(0)}%`, 'Anomaly Score']}
                />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">AI Insights</h3>
          <div className="space-y-4">
            {insights.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No significant AI insights detected. System is operating normally.
              </div>
            ) : (
              insights.map(insight => (
                <div
                  key={insight.id}
                  className={`p-4 rounded-lg ${
                    insight.severity === 'critical' ? 'bg-red-50 dark:bg-red-900/30' : 'bg-yellow-50 dark:bg-yellow-900/30'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {insight.type === 'anomaly' ? (
                        <AlertTriangle className={`h-5 w-5 ${
                          insight.severity === 'critical' ? 'text-red-500' : 'text-yellow-500'
                        }`} />
                      ) : (
                        <TrendingUp className="h-5 w-5 text-purple-500" />
                      )}
                      <h4 className="font-medium">{insight.title}</h4>
                    </div>
                    <span className="text-sm">
                      {(insight.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                  <p className="mt-2 text-sm">{insight.description}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
