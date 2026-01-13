'use client'

import React from 'react';
import { AlertCircle, Shield, Activity, Network, Users, Clock } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Skeleton } from '@/components/ui/skeleton';
import { useAlerts } from '@/services/alert-service';
import { logService } from '@/services/log-service';
import { agentService } from '@/services/agent-service';

interface MetricsData {
  eventsPerSecond: number;
  criticalAlerts: number;
  threatsBlocked: number;
  activeUsers: number;
  networkLoad: number;
  avgResponse: number;
  eventsChange?: number;
  alertsChange?: number;
  threatsChange?: number;
  usersChange?: number;
  networkChange?: number;
  responseChange?: number;
}

export function MetricsGrid() {
  // Fetch alerts for critical alerts count
  const { data: alertsData } = useAlerts({ 
    severity: 'critical',
    limit: 1000,
    status: 'new'
  });

  // Fetch recent logs for events/sec calculation
  const { data: logsData } = useQuery({
    queryKey: ['recent-logs-metrics'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setHours(start.getHours() - 1);
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 10000
      });
    },
    refetchInterval: 30000,
  });

  // Fetch agents for active users count
  const { data: agentsData } = useQuery({
    queryKey: ['agents-metrics'],
    queryFn: () => agentService.getAgents(),
    refetchInterval: 30000,
  });

  const criticalAlerts = alertsData?.items?.length ?? 0;
  const totalLogs = logsData?.totalCount ?? 0;
  const eventsPerSecond = totalLogs > 0 ? Math.round(totalLogs / 3600) : 0;
  const activeAgents = agentsData?.filter(a => a.status === 'Online').length ?? 0;

  const metrics = [
    { 
      label: 'Events/sec', 
      value: eventsPerSecond.toLocaleString(), 
      icon: Activity, 
      change: '+0%', 
      color: 'blue' 
    },
    { 
      label: 'Critical Alerts', 
      value: criticalAlerts.toString(), 
      icon: AlertCircle, 
      change: '+0%', 
      color: 'red' 
    },
    { 
      label: 'Active Agents', 
      value: activeAgents.toString(), 
      icon: Users, 
      change: '+0%', 
      color: 'purple' 
    },
    { 
      label: 'Total Logs (1h)', 
      value: totalLogs.toLocaleString(), 
      icon: Shield, 
      change: '+0%', 
      color: 'green' 
    },
    { 
      label: 'Network Load', 
      value: 'N/A', 
      icon: Network, 
      change: '+0%', 
      color: 'orange' 
    },
    { 
      label: 'Avg Response', 
      value: 'N/A', 
      icon: Clock, 
      change: '+0%', 
      color: 'indigo' 
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {metrics.map((metric) => (
        <div key={metric.label} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">{metric.label}</p>
              <p className="text-2xl font-semibold mt-1 text-gray-900 dark:text-white">{metric.value}</p>
            </div>
            <div className={`rounded-full p-3 bg-${metric.color}-50 dark:bg-${metric.color}-900/20`}>
              <metric.icon className={`w-6 h-6 text-${metric.color}-500`} />
            </div>
          </div>
          {metric.change && (
            <div className="flex items-center mt-4">
              <span className={`text-sm ${metric.change.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
                {metric.change}
              </span>
              <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">vs last hour</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}