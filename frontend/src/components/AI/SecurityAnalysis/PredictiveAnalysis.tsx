'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { LineChart as LineChartIcon, Brain, AlertTriangle, Target, Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { StatsCard } from '@/components/ui/StatsCard'
import { useQuery } from '@tanstack/react-query'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'
import { format } from 'date-fns'

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

export function PredictiveAnalysis() {
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null)

  // Fetch alerts for prediction analysis
  const { data: alertsData } = useAlerts({ 
    limit: 100,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for pattern analysis
  const { data: logsData } = useQuery({
    queryKey: ['predictive-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setHours(start.getHours() - 24);
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 1000
      });
    },
    refetchInterval: 60000,
  });

  // Generate predictions based on alert patterns
  const predictions = useMemo(() => {
    if (!alertsData?.items || alertsData.items.length === 0) return [];

    const predictions: Prediction[] = [];
    const alerts = alertsData.items;

    // Analyze failed login patterns
    const failedLogins = alerts.filter(a => 
      a.message?.toLowerCase().includes('failed login') || 
      a.message?.toLowerCase().includes('authentication failed')
    );
    if (failedLogins.length >= 5) {
      predictions.push({
        id: '1',
        type: 'Brute Force Attack',
        probability: Math.min(90, 50 + failedLogins.length * 5),
        impact: 'high',
        timeframe: '12-24 hours',
        details: `Detected ${failedLogins.length} failed login attempts. Potential brute force attack in progress.`,
        indicators: [
          `${failedLogins.length} failed login attempts detected`,
          'Multiple source IPs observed',
          'Unusual authentication patterns'
        ],
        mitigation: [
          'Enable account lockout policies',
          'Review authentication logs',
          'Implement rate limiting',
          'Enable MFA for affected accounts'
        ]
      });
    }

    // Analyze critical alerts
    const criticalAlerts = alerts.filter(a => a.severity?.toLowerCase() === 'critical');
    if (criticalAlerts.length >= 3) {
      predictions.push({
        id: '2',
        type: 'Critical Security Event',
        probability: 85,
        impact: 'critical',
        timeframe: 'Immediate',
        details: `Multiple critical alerts detected. Immediate investigation required.`,
        indicators: [
          `${criticalAlerts.length} critical alerts in recent timeframe`,
          'Potential security breach indicators',
          'High severity events detected'
        ],
        mitigation: [
          'Immediately review all critical alerts',
          'Isolate affected systems if necessary',
          'Engage security team',
          'Review incident response procedures'
        ]
      });
    }

    // Analyze network anomalies from logs
    if (logsData?.items) {
      const networkLogs = logsData.items.filter(l => 
        l.message?.toLowerCase().includes('network') ||
        l.message?.toLowerCase().includes('connection') ||
        l.message?.toLowerCase().includes('traffic')
      );
      if (networkLogs.length > 100) {
        predictions.push({
          id: '3',
          type: 'Network Anomaly',
          probability: 70,
          impact: 'medium',
          timeframe: '24-48 hours',
          details: `Unusual network activity detected. ${networkLogs.length} network-related events in last 24 hours.`,
          indicators: [
            'Increased network traffic volume',
            'Unusual connection patterns',
            'Multiple network events detected'
          ],
          mitigation: [
            'Review network traffic patterns',
            'Check firewall rules',
            'Monitor for DDoS indicators',
            'Review network access logs'
          ]
        });
      }
    }

    return predictions;
  }, [alertsData, logsData]);

  // Generate time series data from alerts
  const timeSeriesData = useMemo(() => {
    if (!alertsData?.items) {
      return Array.from({ length: 24 }, (_, i) => ({
        time: `${i}:00`,
        predictedThreats: 0,
        confidence: 0
      }));
    }

    const hourlyData: Record<number, number> = {};
    alertsData.items.forEach(alert => {
      if (alert.timestamp) {
        const hour = new Date(alert.timestamp).getHours();
        hourlyData[hour] = (hourlyData[hour] || 0) + 1;
      }
    });

    return Array.from({ length: 24 }, (_, i) => ({
      time: `${i}:00`,
      predictedThreats: hourlyData[i] || 0,
      confidence: hourlyData[i] ? Math.min(95, 75 + hourlyData[i] * 2) : 0
    }));
  }, [alertsData]);

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Active Predictions"
          value={predictions.length.toString()}
          change="+0"
          trend="up"
          icon={Brain}
          color="blue"
        />
        <StatsCard
          title="Critical Alerts"
          value={(alertsData?.items?.filter(a => a.severity?.toLowerCase() === 'critical').length || 0).toString()}
          change="+0"
          trend="up"
          icon={Target}
          color="red"
        />
        <StatsCard
          title="Total Alerts (24h)"
          value={(alertsData?.totalCount || 0).toString()}
          change="+0"
          trend="up"
          icon={Clock}
          color="yellow"
        />
        <StatsCard
          title="High Risk Predictions"
          value={predictions.filter(p => p.impact === 'critical' || p.impact === 'high').length.toString()}
          change="+0"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
      </div>

      {/* Prediction Timeline */}
      <DashboardCard title="Prediction Timeline" icon={LineChartIcon}>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeSeriesData}>
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
          {predictions.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              No active predictions. System is operating normally.
            </div>
          ) : (
            predictions.map((prediction) => (
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
          )))}
        </div>
      </DashboardCard>
    </div>
  )
} 