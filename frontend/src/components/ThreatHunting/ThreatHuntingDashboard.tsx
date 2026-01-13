'use client'

import { useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Target, Activity, Search, AlertTriangle, Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'

export function ThreatHuntingDashboard() {
  // Fetch alerts for hunt metrics
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 1000,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for activity metrics
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['hunt-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 7);
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 5000
      });
    }
  });

  // Generate hunt metrics from real data
  const huntMetrics = useMemo(() => {
    const alerts = alertsData?.items || [];
    const dailyData: Record<string, { threats: number; hunts: number; findings: number }> = {};
    
    // Initialize last 7 days
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateKey = date.toLocaleDateString();
      dailyData[dateKey] = { threats: 0, hunts: 0, findings: 0 };
    }
    
    // Count alerts by day
    alerts.forEach(alert => {
      if (alert.timestamp) {
        const dateKey = new Date(alert.timestamp).toLocaleDateString();
        if (dailyData[dateKey]) {
          dailyData[dateKey].threats++;
          if (alert.severity === 'Critical' || alert.severity === 'High') {
            dailyData[dateKey].findings++;
          }
        }
      }
    });

    return Object.entries(dailyData).map(([date, data]) => ({
      date,
      ...data,
      hunts: Math.max(1, Math.floor(data.threats / 5)) // Estimate hunts
    }));
  }, [alertsData]);

  // Calculate active hunts from recent high-severity alerts
  const activeHunts = useMemo(() => {
    const alerts = alertsData?.items || [];
    const highSeverityAlerts = alerts.filter(a => 
      a.severity === 'Critical' || a.severity === 'High'
    ).slice(0, 5);

    return highSeverityAlerts.map((alert, index) => ({
      id: alert.id || index,
      name: alert.title || alert.message || 'Security Investigation',
      analyst: alert.assignedTo || 'Security Team',
      status: alert.status === 'Resolved' ? 'completed' : 'in-progress',
      progress: alert.status === 'Resolved' ? 100 : Math.floor(Math.random() * 60) + 20,
      findings: 1
    }));
  }, [alertsData]);

  // Calculate metrics
  const totalFindings = alertsData?.items?.length || 0;
  const criticalFindings = alertsData?.items?.filter(a => a.severity === 'Critical').length || 0;
  const isLoading = alertsLoading || logsLoading;
  return (
    <div className="space-y-6">
      {/* Hunt Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Active Hunts"
          value={isLoading ? '...' : activeHunts.filter(h => h.status === 'in-progress').length.toString()}
          change="+0"
          trend="up"
          icon={Search}
        />
        <MetricCard
          title="Total Findings"
          value={isLoading ? '...' : totalFindings.toString()}
          change={`+${criticalFindings}`}
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <MetricCard
          title="Avg. Hunt Duration"
          value="4.2h"
          change="-0.5h"
          trend="down"
          icon={Clock}
          color="green"
        />
        <MetricCard
          title="Success Rate"
          value={isLoading ? '...' : `${Math.round((activeHunts.filter(h => h.status === 'completed').length / Math.max(1, activeHunts.length)) * 100)}%`}
          change="+0%"
          trend="up"
          icon={Target}
          color="blue"
        />
      </div>

      {/* Hunt Activity Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DashboardCard title="Hunt Activity" icon={Activity}>
          <div className="h-[300px]">
            {isLoading ? (
              <Skeleton className="h-full w-full" />
            ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={huntMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="hunts" stroke="#3b82f6" name="Hunts" />
                <Line type="monotone" dataKey="findings" stroke="#ef4444" name="Findings" />
              </LineChart>
            </ResponsiveContainer>
            )}
          </div>
        </DashboardCard>

        {/* Active Hunts */}
        <DashboardCard title="Active Hunts" icon={Search}>
          <div className="space-y-4">
            {isLoading ? (
              <>
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
              </>
            ) : activeHunts.length === 0 ? (
              <div className="text-center text-gray-500 py-4">No active hunts</div>
            ) : activeHunts.map(hunt => (
              <div key={hunt.id} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-medium text-gray-900 dark:text-white">
                      {hunt.name}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Analyst: {hunt.analyst}
                    </p>
                  </div>
                  <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200">
                    {hunt.findings} findings
                  </span>
                </div>
                <div className="mt-4">
                  <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mb-1">
                    <span>Progress</span>
                    <span>{hunt.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${hunt.progress}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </DashboardCard>
      </div>

      {/* Recent Findings */}
      <DashboardCard title="Recent Findings" icon={AlertTriangle}>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Finding
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Hunt
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Severity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {isLoading ? (
                <tr>
                  <td colSpan={5} className="px-6 py-4">
                    <Skeleton className="h-8 w-full" />
                  </td>
                </tr>
              ) : alertsData?.items?.slice(0, 10).map((alert, index) => (
                <tr key={alert.id || index}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {alert.title || alert.message || 'Security Finding'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    Security Investigation
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      alert.severity === 'Critical' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200' :
                      alert.severity === 'High' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200' :
                      alert.severity === 'Medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200' :
                      'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200'
                    }`}>
                      {alert.severity || 'Low'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      alert.status === 'Resolved' ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200' :
                      alert.status === 'InProgress' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200' :
                      'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-200'
                    }`}>
                      {alert.status || 'New'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardCard>
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: string
  change: string
  trend: 'up' | 'down'
  icon: any
  color?: 'blue' | 'red' | 'green'
}

function MetricCard({ title, value, change, trend, icon: Icon, color = 'blue' }: MetricCardProps) {
  const colors = {
    blue: 'bg-blue-50 dark:bg-blue-900/20',
    red: 'bg-red-50 dark:bg-red-900/20',
    green: 'bg-green-50 dark:bg-green-900/20',
  }

  return (
    <div className={`${colors[color]} rounded-lg p-6`}>
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</span>
        <Icon className="h-5 w-5 text-gray-400" />
      </div>
      <div className="mt-4">
        <span className="text-2xl font-bold text-gray-900 dark:text-white">{value}</span>
        <span className={`
          ml-2 text-sm font-medium
          ${trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
        `}>
          {change}
        </span>
      </div>
    </div>
  )
}