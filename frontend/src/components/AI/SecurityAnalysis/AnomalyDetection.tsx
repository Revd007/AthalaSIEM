'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Activity, AlertTriangle, Brain, LineChart, Settings, RefreshCw } from 'lucide-react'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { StatsCard } from '@/components/ui/StatsCard'
import { useQuery } from '@tanstack/react-query'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'

interface AnomalyScore {
  timestamp: string
  score: number
  threshold: number
  category: string
}

export function AnomalyDetection() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [selectedAnomaly, setSelectedAnomaly] = useState<any>(null)

  // Calculate time range
  const getTimeRange = () => {
    const end = new Date();
    const start = new Date();
    switch (selectedTimeRange) {
      case '1h': start.setHours(start.getHours() - 1); break;
      case '24h': start.setHours(start.getHours() - 24); break;
      case '7d': start.setDate(start.getDate() - 7); break;
      case '30d': start.setDate(start.getDate() - 30); break;
    }
    return { start, end };
  };

  const { start, end } = getTimeRange();

  // Fetch alerts as anomalies
  const { data: alertsData, isLoading: alertsLoading, refetch: refetchAlerts } = useAlerts({
    limit: 100,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for anomaly scoring
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['anomaly-logs', selectedTimeRange],
    queryFn: async () => {
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 1000
      });
    },
    refetchInterval: 30000,
  });

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await refetchAlerts()
    setIsRefreshing(false)
  }

  // Convert high-severity alerts to anomalies
  const anomalies = useMemo(() => {
    if (!alertsData?.items) return [];
    
    return alertsData.items
      .filter(a => a.severity?.toLowerCase() === 'high' || a.severity?.toLowerCase() === 'critical')
      .slice(0, 10)
      .map((alert, index) => ({
        id: alert.id,
        type: alert.source || 'Security',
        description: alert.message || alert.title || 'Anomaly detected',
        severity: (alert.severity?.toLowerCase() || 'medium') as 'high' | 'medium',
        timestamp: alert.timestamp || new Date().toISOString(),
        score: alert.severity?.toLowerCase() === 'critical' ? 95 : 75,
        details: {
          alert_id: alert.id,
          source: alert.source,
          agent: alert.agentName || 'Unknown',
          ...alert.details
        }
      }));
  }, [alertsData]);

  // Generate time series data from logs
  const timeSeriesData = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(Date.now() - (23 - i) * 3600000).toISOString(),
        score: 0,
        threshold: 75,
        category: 'normal'
      }));
    }

    const hourlyData: Record<string, { count: number; highSeverity: number }> = {};
    
    logsData.items.forEach(log => {
      if (log.timestamp) {
        const date = new Date(log.timestamp);
        const hourKey = date.toISOString().substring(0, 13) + ':00:00';
        if (!hourlyData[hourKey]) {
          hourlyData[hourKey] = { count: 0, highSeverity: 0 };
        }
        hourlyData[hourKey].count++;
        if (log.severity === 'High' || log.severity === 'Critical') {
          hourlyData[hourKey].highSeverity++;
        }
      }
    });

    // Calculate anomaly scores based on log volume and severity
    return Object.entries(hourlyData)
      .map(([timestamp, data]) => {
        const score = Math.min(100, (data.highSeverity * 20) + (data.count > 50 ? 30 : data.count * 0.6));
        return {
          timestamp,
          score: Math.round(score),
          threshold: 75,
          category: score > 75 ? 'anomaly' : 'normal'
        };
      })
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .slice(-24); // Last 24 hours
  }, [logsData]);

  const isLoading = alertsLoading || logsLoading;
  const anomalyScore = anomalies.length > 0 
    ? Math.round(anomalies.reduce((sum, a) => sum + a.score, 0) / anomalies.length)
    : 0;
  const detectedToday = anomalies.length;

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Anomaly Score"
          value={anomalyScore.toString()}
          change="+0"
          trend="up"
          icon={Activity}
          color="blue"
        />
        <StatsCard
          title="Detected Today"
          value={detectedToday.toString()}
          change="+0"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="High Severity Alerts"
          value={(alertsData?.items?.filter(a => a.severity?.toLowerCase() === 'high' || a.severity?.toLowerCase() === 'critical').length || 0).toString()}
          change="+0"
          trend="up"
          icon={Brain}
          color="red"
        />
        <StatsCard
          title="Total Logs Analyzed"
          value={(logsData?.totalCount || 0).toString()}
          change="+0"
          trend="up"
          icon={LineChart}
          color="blue"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Anomaly Timeline */}
        <div className="lg:col-span-2">
          <DashboardCard title="Anomaly Timeline" icon={Activity}>
            <div className="space-y-4">
              {/* Controls */}
              <div className="flex justify-between">
                <select
                  value={selectedTimeRange}
                  onChange={(e) => setSelectedTimeRange(e.target.value)}
                  className="px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg"
                >
                  <option value="1h">Last Hour</option>
                  <option value="24h">Last 24 Hours</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                </select>
                <button
                  onClick={handleRefresh}
                  className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                >
                  <RefreshCw className={`h-5 w-5 ${isRefreshing ? 'animate-spin' : ''}`} />
                </button>
              </div>

              {/* Chart */}
              <div className="h-[300px]">
                {isLoading ? (
                  <Skeleton className="h-full w-full" />
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeSeriesData}>
                    <defs>
                      <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                              <p className="text-sm text-gray-500 dark:text-gray-400">
                                {new Date(payload[0].payload.timestamp).toLocaleString()}
                              </p>
                              <p className="text-sm font-medium text-gray-900 dark:text-white">
                                Score: {typeof payload[0]?.value === 'number' ? payload[0]?.value.toFixed(2) : 'N/A'}
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="score"
                      stroke="#3b82f6"
                      fillOpacity={1}
                      fill="url(#scoreGradient)"
                    />
                    <Line
                      type="monotone"
                      dataKey="threshold"
                      stroke="#ef4444"
                      strokeDasharray="5 5"
                    />
                  </AreaChart>
                </ResponsiveContainer>
                )}
              </div>
            </div>
          </DashboardCard>
        </div>

        {/* Detected Anomalies */}
        <div className="lg:col-span-1">
          <DashboardCard title="Detected Anomalies" icon={AlertTriangle}>
            <div className="space-y-4">
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-20 w-full" />
                  ))}
                </div>
              ) : anomalies.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  No anomalies detected
                </div>
              ) : (
                anomalies.map((anomaly) => (
                <div
                  key={anomaly.id}
                  onClick={() => setSelectedAnomaly(anomaly)}
                  className={`p-4 rounded-lg cursor-pointer border ${
                    selectedAnomaly?.id === anomaly.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          anomaly.severity === 'high'
                            ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                        }`}>
                          {anomaly.severity}
                        </span>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {anomaly.type}
                        </span>
                      </div>
                      <p className="text-sm text-gray-900 dark:text-white mt-1">
                        {anomaly.description}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-semibold text-gray-900 dark:text-white">
                        {anomaly.score}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        score
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    {new Date(anomaly.timestamp).toLocaleString()}
                  </div>
                </div>
              )))}
            </div>
          </DashboardCard>
        </div>
      </div>

      {/* Model Configuration */}
      <DashboardCard title="Model Configuration" icon={Settings}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white">
              Detection Parameters
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Sensitivity Threshold
                </span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  defaultValue="75"
                  className="w-32"
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Learning Rate
                </span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  defaultValue="85"
                  className="w-32"
                />
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white">
              Model Performance
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  True Positive Rate
                </span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  96.8%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  False Positive Rate
                </span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  3.2%
                </span>
              </div>
            </div>
          </div>
        </div>
      </DashboardCard>
    </div>
  )
} 