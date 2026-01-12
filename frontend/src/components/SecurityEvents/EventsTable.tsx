'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { List, AlertTriangle, ArrowUpRight } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { logService } from '@/services/log-service'
import type { LogEntry } from '@/types/agent'
import { Skeleton } from '@/components/ui/skeleton'

export function EventsTable({ filters }: { filters: any }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['events', filters],
    queryFn: async () => {
      const params: any = {
        limit: 50,
        offset: 0,
        sortField: 'timestamp',
        sortDirection: 'desc'
      };

      if (filters?.severity) params.severity = filters.severity;
      if (filters?.source) params.source = filters.source;
      if (filters?.eventType) params.searchTerm = filters.eventType;
      if (filters?.timeRange) {
        const end = new Date();
        const start = new Date();
        if (filters.timeRange === '24h') {
          start.setHours(start.getHours() - 24);
        } else if (filters.timeRange === '7d') {
          start.setDate(start.getDate() - 7);
        } else if (filters.timeRange === '30d') {
          start.setDate(start.getDate() - 30);
        }
        params.startDate = start.toISOString();
        params.endDate = end.toISOString();
      }

      return await logService.getLogs(params);
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const events = data?.items ?? [];

  return (
    <DashboardCard title="Event Logs" icon={List}>
      <div className="overflow-x-auto">
        {isLoading ? (
          <div className="space-y-2 p-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : error ? (
          <div className="text-center text-red-500 py-4">
            Failed to load events
          </div>
        ) : events.length === 0 ? (
          <div className="text-center text-gray-500 py-4">
            No events found
          </div>
        ) : (
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Source
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Severity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Message
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {events.map((event: LogEntry) => {
                const severity = (event.severity?.toLowerCase() || 'low') as 'critical' | 'high' | 'medium' | 'low';
                return (
                  <tr key={event.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {event.timestamp ? new Date(event.timestamp).toLocaleString() : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {event.source || 'Unknown'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {event.ipAddress || event.agentId || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        severity === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200' :
                        severity === 'high' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200' :
                        severity === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200'
                      }`}>
                        {event.severity || 'low'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900 dark:text-white">
                      {event.message || 'No message'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <button className="text-blue-500 hover:text-blue-600">
                        <ArrowUpRight className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </DashboardCard>
  )
} 