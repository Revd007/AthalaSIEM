'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Globe, Shield, AlertTriangle, RefreshCw, Search, Filter, Download, ExternalLink } from 'lucide-react'
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'
import { StatsCard } from '../SecurityEvents/StatsCard'
import { useQuery } from '@tanstack/react-query'
import { threatIntelligenceService } from '@/services/threat-intelligence'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'

interface ThreatFeed {
  id: string
  name: string
  provider: string
  type: 'ip' | 'domain' | 'hash' | 'url'
  lastUpdate: string
  status: 'active' | 'disabled'
  indicators: number
  matches: number
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981']

export function ThreatIntelligence() {
  const [selectedFeed, setSelectedFeed] = useState<ThreatFeed | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Fetch threat intelligence data
  const { data: threatIntelData, isLoading: threatIntelLoading, refetch: refetchThreatIntel } = useQuery({
    queryKey: ['threat-intelligence'],
    queryFn: () => threatIntelligenceService.getThreatIntelligence(),
    refetchInterval: 300000, // 5 minutes
  });

  // Fetch alerts that might be related to threat intel
  const { data: alertsData } = useAlerts({
    limit: 100,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for threat pattern analysis
  const { data: logsData } = useQuery({
    queryKey: ['threat-logs'],
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

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await refetchThreatIntel()
    setIsRefreshing(false)
  }

  // Generate threat types from alerts/logs
  const threatTypes = useMemo(() => {
    if (!alertsData?.items && !logsData?.items) {
      return [
        { name: 'Malware', value: 0 },
        { name: 'C2', value: 0 },
        { name: 'Phishing', value: 0 },
        { name: 'Ransomware', value: 0 }
      ];
    }

    const types: Record<string, number> = {
      'Malware': 0,
      'C2': 0,
      'Phishing': 0,
      'Ransomware': 0
    };

    // Analyze alerts
    alertsData?.items?.forEach(alert => {
      const message = (alert.message || alert.title || '').toLowerCase();
      if (message.includes('malware') || message.includes('trojan') || message.includes('virus')) {
        types['Malware']++;
      } else if (message.includes('c2') || message.includes('command') || message.includes('control')) {
        types['C2']++;
      } else if (message.includes('phishing') || message.includes('spam')) {
        types['Phishing']++;
      } else if (message.includes('ransomware') || message.includes('encrypt')) {
        types['Ransomware']++;
      }
    });

    // Analyze logs
    logsData?.items?.forEach(log => {
      const message = (log.message || '').toLowerCase();
      if (message.includes('malware') || message.includes('trojan')) {
        types['Malware']++;
      } else if (message.includes('c2') || message.includes('command')) {
        types['C2']++;
      } else if (message.includes('phishing')) {
        types['Phishing']++;
      } else if (message.includes('ransomware')) {
        types['Ransomware']++;
      }
    });

    return Object.entries(types)
      .map(([name, value]) => ({ name, value }))
      .filter(t => t.value > 0)
      .sort((a, b) => b.value - a.value);
  }, [alertsData, logsData]);

  // Generate feeds from threat intel data or use defaults
  const feeds = useMemo(() => {
    if (threatIntelData?.feeds) {
      return threatIntelData.feeds;
    }
    // Return empty array if no data
    return [];
  }, [threatIntelData]);

  const activeFeeds = feeds.filter(f => f.status === 'active').length;
  const totalIndicators = feeds.reduce((sum, f) => sum + (f.indicators || 0), 0);
  const totalMatches = feeds.reduce((sum, f) => sum + (f.matches || 0), 0);

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Active Feeds"
          value={activeFeeds.toString()}
          change="+0"
          trend="up"
          icon={Globe}
        />
        <StatsCard
          title="Total Indicators"
          value={totalIndicators > 1000 ? `${(totalIndicators / 1000).toFixed(1)}K` : totalIndicators.toString()}
          change="+0"
          trend="up"
          icon={Shield}
          color="blue"
        />
        <StatsCard
          title="Matches Today"
          value={totalMatches.toString()}
          change="+0"
          trend="up"
          icon={AlertTriangle}
          color="red"
        />
        <StatsCard
          title="Threat Alerts"
          value={(alertsData?.items?.filter(a => 
            a.message?.toLowerCase().includes('threat') ||
            a.message?.toLowerCase().includes('malware') ||
            a.message?.toLowerCase().includes('attack')
          ).length || 0).toString()}
          change="+0"
          trend="up"
          icon={RefreshCw}
          color="red"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Threat Feeds */}
        <div className="lg:col-span-2">
          <DashboardCard title="Threat Feeds" icon={Globe}>
            <div className="space-y-4">
              {/* Actions */}
              <div className="flex justify-between">
                <div className="flex space-x-2">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search feeds..."
                      className="pl-10 pr-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
                    />
                    <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                  </div>
                  <button className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                    <Filter className="h-5 w-5 text-gray-500" />
                  </button>
                </div>
                <button 
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center"
                  onClick={handleRefresh}
                >
                  {isRefreshing ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4 mr-2" />
                  )}
                  Refresh All
                </button>
              </div>

              {/* Feeds Table */}
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead>
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Feed
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Provider
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Indicators
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Matches
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Last Update
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {threatIntelLoading ? (
                      <tr>
                        <td colSpan={7} className="px-6 py-4">
                          <Skeleton className="h-12 w-full" />
                        </td>
                      </tr>
                    ) : feeds.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-6 py-4 text-center text-gray-500">
                          No threat intelligence feeds configured
                        </td>
                      </tr>
                    ) : (
                      feeds.map(feed => (
                        <tr key={feed.id}>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="font-medium text-gray-900 dark:text-white">
                              {feed.name}
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                            {feed.provider}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm">
                            <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200">
                              {feed.type}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                            {feed.indicators.toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                            {feed.matches.toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                            {new Date(feed.lastUpdate).toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm">
                            <button className="text-blue-500 hover:text-blue-600">
                              <ExternalLink className="h-4 w-4" />
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </DashboardCard>
        </div>

        {/* Threat Distribution */}
        <div className="lg:col-span-1">
          <DashboardCard title="Threat Distribution" icon={AlertTriangle}>
            {threatTypes.length === 0 ? (
              <div className="h-[300px] flex items-center justify-center text-gray-500">
                No threat data available
              </div>
            ) : (
              <>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={threatTypes}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {threatTypes.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4">
                  <div className="grid grid-cols-2 gap-4">
                    {threatTypes.map((type, index) => {
                      const total = threatTypes.reduce((sum, t) => sum + t.value, 0);
                      const percentage = total > 0 ? Math.round((type.value / total) * 100) : 0;
                      return (
                        <div key={type.name} className="flex items-center">
                          <div 
                            className="w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: COLORS[index % COLORS.length] }}
                          />
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {type.name} ({percentage}%)
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </DashboardCard>
        </div>
      </div>
    </div>
  )
} 