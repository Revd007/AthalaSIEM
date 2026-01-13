'use client'

import { useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Activity, AlertTriangle, Shield, Clock } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { StatsCard } from './StatsCard'
import { useQuery } from '@tanstack/react-query'
import { logService } from '@/services/log-service'
import { useAlerts } from '@/services/alert-service'
import { Skeleton } from '@/components/ui/skeleton'

interface SecurityEventsOverviewProps {
  timeRange: string
  filters: any
}

export function SecurityEventsOverview({ timeRange, filters }: SecurityEventsOverviewProps) {
  // Calculate time range
  const getTimeRange = () => {
    const end = new Date();
    const start = new Date();
    switch (timeRange) {
      case '1h': start.setHours(start.getHours() - 1); break;
      case '24h': start.setDate(start.getDate() - 1); break;
      case '7d': start.setDate(start.getDate() - 7); break;
      case '30d': start.setDate(start.getDate() - 30); break;
      default: start.setDate(start.getDate() - 1);
    }
    return { start, end };
  };

  const { start, end } = getTimeRange();

  // Fetch logs
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['security-events', timeRange],
    queryFn: async () => {
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 5000
      });
    },
    refetchInterval: 30000,
  });

  // Fetch alerts
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 1000,
    startTime: start.toISOString(),
    endTime: end.toISOString()
  });

  const isLoading = logsLoading || alertsLoading;

  // Generate chart data from logs
  const chartData = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 24 }, (_, i) => ({
        time: `${i}:00`,
        total: 0, critical: 0, high: 0, medium: 0
      }));
    }

    const hourlyData: Record<string, { total: number; critical: number; high: number; medium: number }> = {};
    
    // Initialize 24 hours
    for (let i = 0; i < 24; i++) {
      hourlyData[`${i}:00`] = { total: 0, critical: 0, high: 0, medium: 0 };
    }

    logsData.items.forEach(log => {
      if (log.timestamp) {
        const hour = new Date(log.timestamp).getHours();
        const key = `${hour}:00`;
        if (hourlyData[key]) {
          hourlyData[key].total++;
          const severity = log.severity?.toLowerCase();
          if (severity === 'critical') hourlyData[key].critical++;
          else if (severity === 'high') hourlyData[key].high++;
          else if (severity === 'medium') hourlyData[key].medium++;
        }
      }
    });

    return Object.entries(hourlyData).map(([time, data]) => ({ time, ...data }));
  }, [logsData]);

  // Calculate statistics
  const totalEvents = logsData?.totalCount || 0;
  const criticalEvents = alertsData?.items?.filter(a => a.severity === 'Critical').length || 0;
  const threatsBlocked = alertsData?.items?.filter(a => a.status === 'Resolved').length || 0;

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard 
          title="Total Events"
          value={isLoading ? '...' : totalEvents > 1000 ? `${(totalEvents / 1000).toFixed(1)}K` : totalEvents.toString()}
          change="+0%"
          icon={Activity}
          trend="up"
        />
        <StatsCard 
          title="Critical Events"
          value={isLoading ? '...' : criticalEvents.toString()}
          change="+0%"
          icon={AlertTriangle}
          trend={criticalEvents > 10 ? 'up' : 'down'}
          color="red"
        />
        <StatsCard 
          title="Avg Response Time"
          value="<1s"
          change="-0%"
          icon={Clock}
          trend="down"
          color="green"
        />
        <StatsCard 
          title="Threats Blocked"
          value={isLoading ? '...' : threatsBlocked.toString()}
          change="+0%"
          icon={Shield}
          trend="up"
          color="blue"
        />
      </div>

      {/* Events Timeline Chart */}
      <DashboardCard title="Events Over Time" icon={Activity}>
        <div className="h-[300px]">
          {isLoading ? (
            <Skeleton className="h-full w-full" />
          ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="critical" 
                stackId="1"
                stroke="#ef4444" 
                fill="#fee2e2" 
                name="Critical"
              />
              <Area 
                type="monotone" 
                dataKey="high" 
                stackId="1"
                stroke="#f97316" 
                fill="#ffedd5" 
                name="High"
              />
              <Area 
                type="monotone" 
                dataKey="medium" 
                stackId="1"
                stroke="#eab308" 
                fill="#fef3c7" 
                name="Medium"
              />
            </AreaChart>
          </ResponsiveContainer>
          )}
        </div>
      </DashboardCard>
    </div>
  )
} 