'use client'

import React, { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { logService } from '@/services/log-service';
import { Skeleton } from '@/components/ui/skeleton';
import { format } from 'date-fns';

export function NetworkTraffic() {
  const { data: logsData, isLoading } = useQuery({
    queryKey: ['network-traffic'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setHours(start.getHours() - 24);
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 10000
      });
    },
    refetchInterval: 60000, // Refresh every minute
  });

  const chartData = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 24 }, (_, i) => ({
        time: format(new Date(Date.now() - (23 - i) * 3600000), 'HH:mm'),
        inbound: 0,
        outbound: 0,
      }));
    }

    // Group logs by hour
    const hourlyData: Record<string, { inbound: number; outbound: number }> = {};
    
    logsData.items.forEach((log) => {
      const hour = new Date(log.timestamp).getHours();
      const hourKey = format(new Date(log.timestamp).setMinutes(0, 0, 0), 'HH:mm');
      
      if (!hourlyData[hourKey]) {
        hourlyData[hourKey] = { inbound: 0, outbound: 0 };
      }
      
      // Simple heuristic: if log has network-related keywords, count as traffic
      const message = (log.message || '').toLowerCase();
      if (message.includes('inbound') || message.includes('received') || message.includes('accept')) {
        hourlyData[hourKey].inbound++;
      } else if (message.includes('outbound') || message.includes('sent') || message.includes('connect')) {
        hourlyData[hourKey].outbound++;
      } else {
        // Default distribution
        hourlyData[hourKey].inbound++;
      }
    });

    // Fill in missing hours with 0
    const result = Array.from({ length: 24 }, (_, i) => {
      const time = format(new Date(Date.now() - (23 - i) * 3600000), 'HH:mm');
      return {
        time,
        inbound: hourlyData[time]?.inbound || 0,
        outbound: hourlyData[time]?.outbound || 0,
      };
    });

    return result;
  }, [logsData]);

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Network Traffic (24h)</h2>
        <Skeleton className="h-80 w-full" />
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Network Traffic (24h)</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="inbound"
              stackId="1"
              stroke="#3b82f6"
              fill="#93c5fd"
              name="Inbound"
            />
            <Area
              type="monotone"
              dataKey="outbound"
              stackId="1"
              stroke="#10b981"
              fill="#6ee7b7"
              name="Outbound"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}