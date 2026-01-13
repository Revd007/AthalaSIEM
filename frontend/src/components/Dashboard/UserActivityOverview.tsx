'use client'

import React from 'react';
import { Users, AlertTriangle, Clock, ArrowRight } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { logService } from '@/services/log-service';
import { Skeleton } from '@/components/ui/skeleton';
import { formatDistanceToNow } from 'date-fns';

interface UserActivityItem {
  id: string;
  user: string;
  action: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high';
  icon: string;
}

const severityColors = {
  low: 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200',
  medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200',
  high: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200',
};

const getSeverity = (logSeverity: string): 'low' | 'medium' | 'high' => {
  const sev = logSeverity?.toLowerCase() || 'low';
  if (sev === 'critical' || sev === 'high') return 'high';
  if (sev === 'medium') return 'medium';
  return 'low';
};

const getIcon = (message: string): string => {
  const msg = message.toLowerCase();
  if (msg.includes('login') || msg.includes('auth')) return '🔑';
  if (msg.includes('file') || msg.includes('access')) return '🔐';
  if (msg.includes('firewall') || msg.includes('network')) return '🛡️';
  return '📋';
};

export function UserActivityOverview() {
  const { data: logsData, isLoading } = useQuery({
    queryKey: ['user-activity'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setHours(start.getHours() - 24);
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 10,
        sortField: 'timestamp',
        sortDirection: 'desc'
      });
    },
    refetchInterval: 30000,
  });

  const activities: UserActivityItem[] = (logsData?.items || [])
    .filter(log => log.username)
    .slice(0, 5)
    .map((log): UserActivityItem => ({
      id: log.id,
      user: log.username || 'Unknown',
      action: log.message || 'Activity detected',
      timestamp: log.timestamp,
      severity: getSeverity(log.severity?.toString() || 'low'),
      icon: getIcon(log.message || '')
    }));

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Users className="h-5 w-5 text-blue-500" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent User Activity</h3>
        </div>
        <button className="text-sm text-blue-600 hover:text-blue-700 flex items-center">
          View All
          <ArrowRight className="h-4 w-4 ml-1" />
        </button>
      </div>

      <div className="space-y-3">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
        ) : activities.length === 0 ? (
          <div className="text-center text-gray-500 py-4">No recent user activity</div>
        ) : (
          activities.map((activity) => (
            <div
              key={activity.id}
              className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
            >
              <div className="flex-shrink-0 text-xl">{activity.icon}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {activity.user}
                  </p>
                  <span className={`px-2 py-1 text-xs rounded-full ${severityColors[activity.severity]}`}>
                    {activity.severity}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-300">{activity.action}</p>
                <div className="flex items-center mt-1 text-xs text-gray-500 dark:text-gray-400">
                  <Clock className="h-3 w-3 mr-1" />
                  {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}