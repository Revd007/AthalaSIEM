'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Shield, AlertTriangle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useAlerts } from '@/services/alert-service'
import { useMemo } from 'react'
import { Skeleton } from '@/components/ui/skeleton'

export function ThreatHunting() {
  const { data: alertsData, isLoading } = useAlerts({
    limit: 50,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Generate threat type distribution from alerts
  const threatData = useMemo(() => {
    if (!alertsData?.items) return [];

    const types: Record<string, number> = {};
    alertsData.items.forEach(alert => {
      const message = (alert.message || alert.title || '').toLowerCase();
      if (message.includes('malware') || message.includes('trojan')) {
        types['Malware'] = (types['Malware'] || 0) + 1;
      } else if (message.includes('phishing') || message.includes('spam')) {
        types['Phishing'] = (types['Phishing'] || 0) + 1;
      } else if (message.includes('exfil') || message.includes('data transfer')) {
        types['Data Exfil'] = (types['Data Exfil'] || 0) + 1;
      } else if (message.includes('lateral') || message.includes('movement')) {
        types['Lateral Movement'] = (types['Lateral Movement'] || 0) + 1;
      } else if (message.includes('privilege') || message.includes('escalation')) {
        types['Privilege Esc'] = (types['Privilege Esc'] || 0) + 1;
      }
    });

    return Object.entries(types)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5);
  }, [alertsData]);

  const recentFindings = alertsData?.items?.slice(0, 3) || [];

  return (
    <DashboardCard title="Threat Hunting Overview" icon={Shield}>
      <div className="space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Active Hunts</div>
            <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">
              {isLoading ? '...' : (alertsData?.items?.filter(a => a.status === 'new' || a.status === 'in_progress').length || 0)}
            </div>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Findings Today</div>
            <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">
              {isLoading ? '...' : (alertsData?.totalCount || 0)}
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="h-[200px]">
          {isLoading ? (
            <Skeleton className="h-full w-full" />
          ) : threatData.length === 0 ? (
            <div className="h-full flex items-center justify-center text-gray-500">
              No threat data available
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={threatData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Recent Findings */}
        <div>
          <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
            Recent Findings
          </h3>
          <div className="space-y-2">
            {isLoading ? (
              <Skeleton className="h-16 w-full" />
            ) : recentFindings.length === 0 ? (
              <div className="text-center text-gray-500 py-4">No recent findings</div>
            ) : (
              recentFindings.map((alert) => (
                <div key={alert.id} className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <div className="flex items-center">
                    <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2" />
                    <span className="text-sm text-yellow-800 dark:text-yellow-200">
                      {alert.title || alert.message || 'Threat detected'}
                    </span>
                  </div>
                  <span className="text-xs text-yellow-600 dark:text-yellow-300 mt-1 block">
                    {new Date(alert.timestamp).toLocaleString()}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </DashboardCard>
  )
} 