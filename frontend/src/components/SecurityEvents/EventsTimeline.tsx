'use client'

import { useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'

export function EventsTimeline() {
  // Fetch logs for timeline
  const { data: logsData, isLoading } = useQuery({
    queryKey: ['events-timeline'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 1);
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 5000
      });
    },
    refetchInterval: 30000,
  });

  // Generate timeline data
  const timelineData = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 24 }, (_, i) => ({
        time: `${i}:00`,
        events: 0,
        anomalies: 0
      }));
    }

    const hourlyData: Record<string, { events: number; anomalies: number }> = {};
    
    for (let i = 0; i < 24; i++) {
      hourlyData[`${i}:00`] = { events: 0, anomalies: 0 };
    }

    logsData.items.forEach(log => {
      if (log.timestamp) {
        const hour = new Date(log.timestamp).getHours();
        const key = `${hour}:00`;
        if (hourlyData[key]) {
          hourlyData[key].events++;
          if (log.severity === 'High' || log.severity === 'Critical') {
            hourlyData[key].anomalies++;
          }
        }
      }
    });

    return Object.entries(hourlyData).map(([time, data]) => ({ time, ...data }));
  }, [logsData]);

  return (
    <DashboardCard title="Events Timeline" icon={Clock}>
      <div className="h-[300px]">
        {isLoading ? (
          <Skeleton className="h-full w-full" />
        ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time"
              tick={{ fontSize: 12 }}
              tickLine={false}
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: 'none',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
              }}
            />
            <Line
              type="monotone"
              dataKey="events"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Events"
            />
            <Line
              type="monotone"
              dataKey="anomalies"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              name="Anomalies"
            />
          </LineChart>
        </ResponsiveContainer>
        )}
      </div>

      {/* Timeline Legend */}
      <div className="flex items-center justify-center space-x-6 mt-4">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-blue-500 mr-2" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Normal Events</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-2" />
          <span className="text-sm text-gray-600 dark:text-gray-400">Anomalies</span>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="text-sm text-gray-500 dark:text-gray-400">Peak Events</div>
          <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">
            {timelineData.length > 0 ? Math.max(...timelineData.map(d => d.events)) : 0}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            at {timelineData.length > 0 ? timelineData[timelineData.findIndex(d => d.events === Math.max(...timelineData.map(d => d.events)))]?.time || 'N/A' : 'N/A'}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="text-sm text-gray-500 dark:text-gray-400">Total Anomalies</div>
          <div className="text-2xl font-semibold text-gray-900 dark:text-white mt-1">
            {timelineData.reduce((acc, curr) => acc + curr.anomalies, 0)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            in last 24 hours
          </div>
        </div>
      </div>
    </DashboardCard>
  )
} 