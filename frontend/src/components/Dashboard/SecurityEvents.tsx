'use client'

import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { logService } from '@/services/log-service';
import { Skeleton } from '@/components/ui/skeleton';

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899'];

export function SecurityEvents() {
  // Check if user is authenticated before making API calls
  const isAuthenticated = typeof window !== 'undefined' && !!localStorage.getItem('token');
  
  const { data: logsData, isLoading } = useQuery({
    queryKey: ['security-events-distribution'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 7); // Last 7 days
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 10000
      });
    },
    enabled: isAuthenticated, // Only run query if authenticated
    refetchInterval: 60000, // Refresh every minute
  });

  const distributionData = useMemo(() => {
    if (!logsData?.items) return [];

    const categoryCounts: Record<string, number> = {};
    
    logsData.items.forEach((log) => {
      const category = log.source || log.category || 'Other';
      categoryCounts[category] = (categoryCounts[category] || 0) + 1;
    });

    // Get top 6 categories
    const sorted = Object.entries(categoryCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 6)
      .map(([name, value]) => ({ name, value }));

    return sorted;
  }, [logsData]);

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Security Events Distribution</h2>
        <Skeleton className="h-80 w-full" />
      </div>
    );
  }

  if (distributionData.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Security Events Distribution</h2>
        <div className="h-80 flex items-center justify-center text-gray-500">
          No events data available
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Security Events Distribution</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={distributionData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
            >
              {distributionData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}