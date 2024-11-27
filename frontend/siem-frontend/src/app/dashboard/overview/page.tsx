'use client'

import { Suspense } from 'react'
import { Card } from '../../../components/ui/card'
import { SystemMonitor } from '../../../components/dashboard/system-monitor'
import { EventsOverview } from '../../../components/dashboard/events/events-overview'
import { AIInsightsCard } from '../../../components/ai/ai-insights-card'
import { LoadingSkeleton } from '../../../components/ui/loading-skeleton'
import { EventsChart } from '../../../components/dashboard/events/events-chart'
import { useQuery } from '@tanstack/react-query'
import { dashboardService } from '../../../services/dashboard-service'

export default function DashboardOverview() {
  const { data: systemMetrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: () => dashboardService.getSystemMetrics(),
    refetchInterval: 5000,
  });

  const { data: eventsData, isLoading: eventsLoading } = useQuery({
    queryKey: ['events-overview'],
    queryFn: () => dashboardService.getEventsOverview(),
    refetchInterval: 30000,
  });

  const { data: recentEvents, isLoading: eventsRecentLoading } = useQuery({
    queryKey: ['recent-events'],
    queryFn: () => dashboardService.getRecentEvents(20),
    refetchInterval: 30000,
  });

  return (
    <div className="min-h-screen bg-gray-50/30">
      <div className="max-w-[2000px] mx-auto p-4 sm:p-6 lg:p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-gray-900">System Overview</h1>
          <p className="mt-1 text-sm text-gray-500">Real-time monitoring and analysis</p>
        </div>

        {/* System Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100">CPU Usage</p>
                <h3 className="text-3xl font-bold mt-1">{systemMetrics?.cpu || '0'}%</h3>
              </div>
              <div className="p-3 bg-blue-400/30 rounded-lg">
                {/* Icon here */}
              </div>
            </div>
            <div className="mt-4 h-2 bg-blue-400/30 rounded-full">
              <div 
                className="h-2 bg-white rounded-full transition-all duration-500"
                style={{ width: `${systemMetrics?.cpu || 0}%` }}
              />
            </div>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500 to-purple-600 text-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100">Memory Usage</p>
                <h3 className="text-3xl font-bold mt-1">{systemMetrics?.memory || '0'}%</h3>
              </div>
              <div className="p-3 bg-purple-400/30 rounded-lg">
                {/* Icon here */}
              </div>
            </div>
            <div className="mt-4 h-2 bg-purple-400/30 rounded-full">
              <div 
                className="h-2 bg-white rounded-full transition-all duration-500"
                style={{ width: `${systemMetrics?.memory || 0}%` }}
              />
            </div>
          </Card>

          <Card className="bg-gradient-to-br from-emerald-500 to-emerald-600 text-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-emerald-100">Storage</p>
                <h3 className="text-3xl font-bold mt-1">{systemMetrics?.storage || '0'}%</h3>
              </div>
              <div className="p-3 bg-emerald-400/30 rounded-lg">
                {/* Icon here */}
              </div>
            </div>
            <div className="mt-4 h-2 bg-emerald-400/30 rounded-full">
              <div 
                className="h-2 bg-white rounded-full transition-all duration-500"
                style={{ width: `${systemMetrics?.storage || 0}%` }}
              />
            </div>
          </Card>

          <Card className="bg-gradient-to-br from-amber-500 to-amber-600 text-white p-6 rounded-xl shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-amber-100">Network</p>
                <h3 className="text-3xl font-bold mt-1">{systemMetrics?.network || '0'} MB/s</h3>
              </div>
              <div className="p-3 bg-amber-400/30 rounded-lg">
                {/* Icon here */}
              </div>
            </div>
            <div className="mt-4 h-2 bg-amber-400/30 rounded-full">
              <div 
                className="h-2 bg-white rounded-full transition-all duration-500"
                style={{ width: `${systemMetrics?.networkUsage || 0}%` }}
              />
            </div>
          </Card>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <Card className="bg-white p-6 rounded-xl shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Event Overview</h3>
                <p className="text-sm text-gray-500 mt-1">Event distribution over time</p>
              </div>
              <select className="text-sm border rounded-lg px-3 py-2">
                <option>Last 24 hours</option>
                <option>Last 7 days</option>
                <option>Last 30 days</option>
              </select>
            </div>
            <Suspense fallback={<LoadingSkeleton />}>
              <EventsOverview />
            </Suspense>
          </Card>

          <Suspense fallback={<LoadingSkeleton rows={4} />}>
            <AIInsightsCard />
          </Suspense>
        </div>

        {/* Recent Events Table */}
        <Card className="bg-white rounded-xl shadow-sm overflow-hidden">
          <div className="p-6 border-b">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Recent Events</h3>
                <p className="text-sm text-gray-500 mt-1">Latest security events and alerts</p>
              </div>
              <button className="px-4 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors text-sm font-medium">
                View All
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Severity</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {/* Event rows here */}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  )
}