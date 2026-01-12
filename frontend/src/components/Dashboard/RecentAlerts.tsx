'use client'

import React from 'react';
import { AlertCircle, Shield, AlertTriangle, Info, AlertOctagon } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Alert, PaginatedResult } from '@/types/alert';
import { Skeleton } from '@/components/ui/skeleton';

const severityIcon = {
  low: Info,
  medium: AlertCircle,
  high: AlertTriangle,
  critical: AlertOctagon
} as const;

const severityColor = {
  low: 'blue',
  medium: 'yellow',
  high: 'orange',
  critical: 'red'
} as const;

function formatTimeAgo(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
}

export function RecentAlerts() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['recent-alerts'],
    queryFn: async () => {
      const queryParams = new URLSearchParams({
        limit: '5',
        offset: '0',
        sortField: 'Timestamp',
        sortDirection: 'desc'
      });
      
      const { data } = await api.get<PaginatedResult<Alert>>(`/api/alerts?${queryParams.toString()}`);
      return data?.items ?? [];
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
        <div className="text-center text-gray-500 py-4">
          Failed to load alerts
        </div>
      </div>
    );
  }

  const alerts = data ?? [];

  if (alerts.length === 0) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
        <div className="text-center text-gray-500 py-4">
          No recent alerts
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
      <div className="space-y-4">
        {alerts.map((alert) => {
          const severity = (alert.severity?.toLowerCase() || 'low') as 'low' | 'medium' | 'high' | 'critical';
          const Icon = severityIcon[severity] || Info;
          const color = severityColor[severity] || 'blue';
          
          return (
            <div
              key={alert.id}
              className={`flex items-start space-x-4 p-4 bg-${color}-50 rounded-lg`}
            >
              <Icon className={`w-5 h-5 text-${color}-500 mt-1`} />
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <h3 className="font-medium">{alert.title || alert.message || 'Alert'}</h3>
                  <span className={`text-xs px-2 py-1 rounded-full bg-${color}-100 text-${color}-800`}>
                    {alert.severity}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">{alert.description || alert.message || ''}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {alert.timestamp ? formatTimeAgo(alert.timestamp) : 'Unknown time'}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}