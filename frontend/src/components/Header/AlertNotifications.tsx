'use client'

import React, { useState } from 'react';
import { Bell, X } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Alert, PaginatedResult } from '@/types/alert';
import { Skeleton } from '@/components/ui/skeleton';

export function AlertNotifications() {
  const [isOpen, setIsOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ['alert-notifications'],
    queryFn: async () => {
      const queryParams = new URLSearchParams({
        limit: '10',
        offset: '0',
        sortField: 'Timestamp',
        sortDirection: 'desc',
        status: 'new'
      });
      
      const { data } = await api.get<PaginatedResult<Alert>>(`/api/alerts?${queryParams.toString()}`);
      return data?.items ?? [];
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const markAsReadMutation = useMutation({
    mutationFn: async (alertId: string) => {
      await api.patch(`/api/alerts/${alertId}/status`, {
        status: 'in_progress',
        updatedBy: 'system',
        updatedAt: new Date().toISOString()
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alert-notifications'] });
    }
  });

  const alerts = data ?? [];
  const unreadCount = alerts.filter(alert => alert.status === 'new').length;

  const markAsRead = (alertId: string) => {
    markAsReadMutation.mutate(alertId);
  };

  return (
    <div className="relative">
      <button 
        className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 relative"
        onClick={() => setIsOpen(!isOpen)}
      >
        <Bell className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 h-4 w-4 bg-red-500 rounded-full text-xs text-white flex items-center justify-center">
            {unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-white dark:bg-gray-800 rounded-lg shadow-lg ring-1 ring-black ring-opacity-5">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">Notifications</h3>
              <button 
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-500"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-20 w-full" />
                  ))}
                </div>
              ) : alerts.length === 0 ? (
                <div className="text-center text-gray-500 py-4">
                  No new alerts
                </div>
              ) : (
                alerts.map(alert => {
                  const isUnread = alert.status === 'new';
                  const severity = (alert.severity?.toLowerCase() || 'low') as 'critical' | 'high' | 'medium' | 'low';
                  
                  return (
                    <div 
                      key={alert.id}
                      className={`p-3 rounded-lg cursor-pointer ${
                        isUnread ? 'bg-blue-50 dark:bg-blue-900/20' : 'bg-gray-50 dark:bg-gray-700'
                      }`}
                      onClick={() => markAsRead(alert.id)}
                    >
                      <div className="flex items-center justify-between">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          severity === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200' :
                          severity === 'high' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200' :
                          'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                        }`}>
                          {alert.severity}
                        </span>
                        <span className="text-xs text-gray-500">
                          {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : ''}
                        </span>
                      </div>
                      <h4 className="font-medium text-gray-900 dark:text-white mt-2">
                        {alert.title || alert.message || 'Alert'}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                        {alert.description || alert.message || ''}
                      </p>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 