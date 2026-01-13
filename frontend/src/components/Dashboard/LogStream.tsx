'use client'

import React from 'react';
import { Clock, Server } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { logService } from '@/services/log-service';
import { Skeleton } from '@/components/ui/skeleton';
import type { LogEntry } from '@/types/agent';

const getSeverityClasses = (severity: string) => {
  const sev = severity?.toLowerCase() || 'low';
  if (sev === 'critical' || sev === 'high') return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200';
  if (sev === 'medium' || sev === 'warning') return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200';
  return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200';
};

export function LogStream() {
  const { data: logsData, isLoading } = useQuery({
    queryKey: ['live-log-stream'],
    queryFn: async () => {
      return await logService.getLogs({
        limit: 10,
        sortField: 'timestamp',
        sortDirection: 'desc'
      });
    },
    refetchInterval: 5000, // Refresh every 5 seconds for live stream
  });

  const logs = logsData?.items ?? [];

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Live Log Stream</h2>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (logs.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
        <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Live Log Stream</h2>
        <div className="text-center text-gray-500 py-8">
          No logs available
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Live Log Stream</h2>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {logs.map((log: LogEntry) => (
          <div key={log.id} className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <Server className="w-5 h-5 text-gray-400 mt-1" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-900 dark:text-white">{log.source || 'Unknown'}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${getSeverityClasses(log.severity?.toString() || 'low')}`}>
                  {log.severity || 'low'}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">{log.message || 'No message'}</p>
              <div className="flex items-center space-x-2 mt-1">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {log.timestamp ? new Date(log.timestamp).toLocaleString() : 'Unknown time'}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}