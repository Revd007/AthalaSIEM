'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Users, Activity, AlertTriangle, Brain, Network, Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { StatsCard } from '@/components/ui/StatsCard'
import { useQuery } from '@tanstack/react-query'
import { logService } from '@/services/log-service'
import { useAlerts } from '@/services/alert-service'
import { Skeleton } from '@/components/ui/skeleton'

interface UserBehavior {
  id: string
  username: string
  department: string
  riskScore: number
  anomalyCount: number
  lastActivity: string
  behaviors: {
    timestamp: string
    activity: string
    risk: 'high' | 'medium' | 'low'
    details: string
  }[]
}

export function BehavioralAnalytics() {
  const [selectedUser, setSelectedUser] = useState<UserBehavior | null>(null)
  const [timeRange, setTimeRange] = useState('24h')

  // Fetch logs for behavioral analysis
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['behavioral-logs', timeRange],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      switch (timeRange) {
        case '1h': start.setHours(start.getHours() - 1); break;
        case '24h': start.setDate(start.getDate() - 1); break;
        case '7d': start.setDate(start.getDate() - 7); break;
        default: start.setDate(start.getDate() - 1);
      }
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 2000
      });
    },
    refetchInterval: 60000,
  });

  // Fetch alerts
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 100,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  const isLoading = logsLoading || alertsLoading;

  // Generate timeline data from logs
  const timelineData = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 24 }, (_, i) => ({
        time: new Date(Date.now() - i * 3600000).toISOString(),
        normalActivity: 0,
        anomalousActivity: 0
      })).reverse();
    }

    const hourlyData: Record<string, { normalActivity: number; anomalousActivity: number }> = {};
    
    for (let i = 23; i >= 0; i--) {
      const date = new Date();
      date.setHours(date.getHours() - i);
      const key = date.toISOString().substring(0, 13);
      hourlyData[key] = { normalActivity: 0, anomalousActivity: 0 };
    }

    logsData.items.forEach(log => {
      if (log.timestamp) {
        const key = log.timestamp.substring(0, 13);
        if (hourlyData[key]) {
          if (log.severity === 'High' || log.severity === 'Critical') {
            hourlyData[key].anomalousActivity++;
          } else {
            hourlyData[key].normalActivity++;
          }
        }
      }
    });

    return Object.entries(hourlyData).map(([time, data]) => ({
      time: new Date(time + ':00:00').toISOString(),
      ...data
    }));
  }, [logsData]);

  // Generate user behaviors from logs
  const users = useMemo(() => {
    if (!logsData?.items) return [];

    const userMap: Record<string, UserBehavior> = {};
    
    logsData.items.forEach((log, index) => {
      const username = log.username || log.source || `user-${index % 5}`;
      
      if (!userMap[username]) {
        userMap[username] = {
          id: username,
          username,
          department: 'Unknown',
          riskScore: 0,
          anomalyCount: 0,
          lastActivity: log.timestamp || new Date().toISOString(),
          behaviors: []
        };
      }

      const user = userMap[username];
      
      if (log.severity === 'High' || log.severity === 'Critical') {
        user.anomalyCount++;
        user.riskScore = Math.min(100, user.riskScore + 10);
        user.behaviors.push({
          timestamp: log.timestamp || new Date().toISOString(),
          activity: log.category || 'Security Event',
          risk: log.severity === 'Critical' ? 'high' : 'medium',
          details: log.message || 'Anomalous activity detected'
        });
      }

      if (new Date(log.timestamp || 0) > new Date(user.lastActivity)) {
        user.lastActivity = log.timestamp || new Date().toISOString();
      }
    });

    return Object.values(userMap)
      .filter(u => u.anomalyCount > 0)
      .sort((a, b) => b.riskScore - a.riskScore)
      .slice(0, 10);
  }, [logsData]);

  const totalUsersMonitored = users.length || 0;
  const highRiskUsers = users.filter(u => u.riskScore > 70).length;
  const totalAnomalies = users.reduce((acc, u) => acc + u.anomalyCount, 0);
  const avgRiskScore = users.length > 0 ? Math.round(users.reduce((acc, u) => acc + u.riskScore, 0) / users.length) : 0;

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Users Monitored"
          value={isLoading ? '...' : totalUsersMonitored.toString()}
          change="+0"
          trend="up"
          icon={Users}
        />
        <StatsCard
          title="Anomalies Today"
          value={isLoading ? '...' : totalAnomalies.toString()}
          change="+0"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Avg Risk Score"
          value={isLoading ? '...' : avgRiskScore.toString()}
          change="+0"
          trend="up"
          icon={Activity}
          color="yellow"
        />
        <StatsCard
          title="High Risk Users"
          value={isLoading ? '...' : highRiskUsers.toString()}
          change="+0"
          trend="up"
          icon={Brain}
          color="red"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Timeline */}
        <div className="lg:col-span-2">
          <DashboardCard title="User Activity Timeline" icon={Activity}>
            <div className="space-y-4">
              {/* Time Range Selector */}
              <div className="flex justify-end">
                <select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  className="px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg"
                >
                  <option value="1h">Last Hour</option>
                  <option value="24h">Last 24 Hours</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                </select>
              </div>

              {/* Activity Chart */}
              <div className="h-[300px]">
                {logsLoading ? (
                  <Skeleton className="h-full w-full" />
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timelineData}>
                    <defs>
                      <linearGradient id="normalActivity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="anomalousActivity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time"
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                              <p className="text-sm text-gray-500 dark:text-gray-400">
                                {new Date(payload[0].payload.time).toLocaleString()}
                              </p>
                              <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                                Normal: {payload[0].value}
                              </p>
                              <p className="text-sm font-medium text-red-600 dark:text-red-400">
                                Anomalous: {payload[1].value}
                              </p>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="normalActivity"
                      stroke="#3b82f6"
                      fillOpacity={1}
                      fill="url(#normalActivity)"
                    />
                    <Area
                      type="monotone"
                      dataKey="anomalousActivity"
                      stroke="#ef4444"
                      fillOpacity={1}
                      fill="url(#anomalousActivity)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
                )}
              </div>
            </div>
          </DashboardCard>
        </div>

        {/* User Risk Scores */}
        <div className="lg:col-span-1">
          <DashboardCard title="User Risk Analysis" icon={Users}>
            <div className="space-y-4">
              {isLoading ? (
                <>
                  <Skeleton className="h-24 w-full" />
                  <Skeleton className="h-24 w-full" />
                </>
              ) : users.length === 0 ? (
                <div className="text-center text-gray-500 py-4">No user risk data available</div>
              ) : users.map((user) => (
                <div
                  key={user.id}
                  onClick={() => setSelectedUser(user)}
                  className={`p-4 rounded-lg cursor-pointer border ${
                    selectedUser?.id === user.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {user.username}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {user.department}
                        </span>
                      </div>
                      <div className="mt-2 flex items-center space-x-4">
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          Risk Score: {user.riskScore}
                        </span>
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          Anomalies: {user.anomalyCount}
                        </span>
                      </div>
                    </div>
                    <div className="text-right text-xs text-gray-500 dark:text-gray-400">
                      Last active: {new Date(user.lastActivity).toLocaleTimeString()}
                    </div>
                  </div>

                  {selectedUser?.id === user.id && (
                    <div className="mt-4 space-y-3">
                      <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                        Recent Behaviors
                      </h4>
                      {user.behaviors.map((behavior, index) => (
                        <div
                          key={index}
                          className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                        >
                          <div className="flex items-center justify-between">
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              behavior.risk === 'high'
                                ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'
                                : behavior.risk === 'medium'
                                ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                                : 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200'
                            }`}>
                              {behavior.risk}
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {new Date(behavior.timestamp).toLocaleString()}
                            </span>
                          </div>
                          <p className="mt-2 text-sm text-gray-900 dark:text-white">
                            {behavior.activity}
                          </p>
                          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                            {behavior.details}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  )
} 