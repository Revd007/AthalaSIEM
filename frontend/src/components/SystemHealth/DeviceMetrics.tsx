'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { 
  Cpu, 
  CircuitBoard,
  HardDrive, 
  Activity,
  Thermometer,
  Network
} from 'lucide-react'
import type { DeviceHealth } from '@/types/system-health'

interface DeviceMetricsProps {
  deviceId: string
}

function MetricCard({ 
  title, 
  value, 
  icon: Icon, 
  secondaryValue,
  progress 
}: { 
  title: string
  value: string
  icon: any
  secondaryValue?: string
  progress?: number
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">
          {title}
        </CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {secondaryValue && (
          <p className="text-xs text-muted-foreground">
            {secondaryValue}
          </p>
        )}
        {typeof progress === 'number' && (
          <div className="mt-4">
            <Progress value={progress} className="h-2" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

'use client'

import { useQuery } from '@tanstack/react-query'
import { agentService } from '@/services/agent-service'
import { Skeleton } from '@/components/ui/skeleton'
import type { Agent } from '@/types/agent'

function formatBytes(bytes: number): string {
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  if (bytes === 0) return '0 B'
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
}

export function DeviceMetrics({ deviceId }: DeviceMetricsProps) {
  const { data: agent, isLoading } = useQuery({
    queryKey: ['agent', deviceId],
    queryFn: () => agentService.getAgentStatus(deviceId),
    enabled: !!deviceId,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const { data: metricsData } = useQuery({
    queryKey: ['agent-metrics', deviceId],
    queryFn: () => {
      const end = new Date();
      const start = new Date();
      start.setHours(start.getHours() - 1);
      return agentService.getAgentMetrics(deviceId, { start, end });
    },
    enabled: !!deviceId && !!agent,
    refetchInterval: 30000,
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Performance Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="text-center text-gray-500 py-4">
        Agent not found
      </div>
    );
  }

  // Use agent metrics or fallback to defaults
  const cpuUsagePercent = agent.cpuUsage ?? 0;
  const memoryUsagePercent = agent.memoryUsage ?? 0;
  const diskUsagePercent = agent.diskUsage ?? 0;
  
  // Get latest metrics from time series if available
  const latestMetrics = metricsData && metricsData.length > 0 ? metricsData[metricsData.length - 1] : null;

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Performance Metrics</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          title="CPU Usage"
          value={`${cpuUsagePercent.toFixed(1)}%`}
          icon={Cpu}
          secondaryValue={latestMetrics?.cpuUsage ? `Current: ${latestMetrics.cpuUsage.toFixed(1)}%` : 'No recent data'}
          progress={cpuUsagePercent}
        />
        
        <MetricCard
          title="Memory Usage"
          value={`${memoryUsagePercent.toFixed(1)}%`}
          icon={CircuitBoard}
          secondaryValue={latestMetrics?.memoryUsage ? `Current: ${latestMetrics.memoryUsage.toFixed(1)}%` : 'No recent data'}
          progress={memoryUsagePercent}
        />
        
        <MetricCard
          title="Disk Usage"
          value={`${diskUsagePercent.toFixed(1)}%`}
          icon={HardDrive}
          secondaryValue={latestMetrics?.diskUsage ? `Current: ${latestMetrics.diskUsage.toFixed(1)}%` : 'No recent data'}
          progress={diskUsagePercent}
        />
        
        {latestMetrics?.networkIn !== undefined && (
          <MetricCard
            title="Network In"
            value={formatBytes(latestMetrics.networkIn)}
            icon={Network}
            secondaryValue="Last hour"
          />
        )}
        
        {latestMetrics?.networkOut !== undefined && (
          <MetricCard
            title="Network Out"
            value={formatBytes(latestMetrics.networkOut)}
            icon={Activity}
            secondaryValue="Last hour"
          />
        )}
      </div>
    </div>
  )
} 