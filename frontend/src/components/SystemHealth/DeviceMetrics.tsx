'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  Cpu, 
  CircuitBoard,
  HardDrive, 
  Activity,
  Thermometer,
  Network,
  Server,
  Clock,
  Wifi,
  WifiOff,
  Info,
  Monitor,
  Calendar
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { agentService } from '@/services/agent-service'
import { Skeleton } from '@/components/ui/skeleton'
import type { DeviceHealth } from '@/types/system-health'
import type { Agent, AgentStatus } from '@/types/agent'

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
      <CardHeader className="flex flex-row items-center justify-between space-y-0 p-3 sm:p-6 pb-1 sm:pb-2">
        <CardTitle className="text-xs sm:text-sm font-medium">
          {title}
        </CardTitle>
        <Icon className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent className="p-3 sm:p-6 pt-0">
        <div className="text-lg sm:text-2xl font-bold">{value}</div>
        {secondaryValue && (
          <p className="text-[10px] sm:text-xs text-muted-foreground truncate">
            {secondaryValue}
          </p>
        )}
        {typeof progress === 'number' && (
          <div className="mt-2 sm:mt-4">
            <Progress value={progress} className="h-1.5 sm:h-2" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

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
    refetchInterval: 30000,
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
      <div className="space-y-3 sm:space-y-4">
        <h2 className="text-base sm:text-lg font-semibold">Performance Metrics</h2>
        <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-3 gap-2 sm:gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Skeleton key={i} className="h-24 sm:h-32 w-full" />
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

  const isOnline = agent.status === 'Online' || agent.status === 'Active';
  const statusColor = isOnline ? 'bg-green-500' : 'bg-gray-400';

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Agent Overview Card */}
      <Card>
        <CardHeader className="p-3 sm:p-6 pb-2 sm:pb-3">
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-sm sm:text-lg flex items-center gap-1.5 sm:gap-2">
              <Server className="h-4 w-4 sm:h-5 sm:w-5" />
              Agent Details
            </CardTitle>
            <Badge 
              variant={isOnline ? 'default' : 'secondary'}
              className={`text-[10px] sm:text-xs ${isOnline ? 'bg-green-500 hover:bg-green-600' : 'bg-gray-500'}`}
            >
              {isOnline ? <Wifi className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-0.5 sm:mr-1" /> : <WifiOff className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-0.5 sm:mr-1" />}
              {agent.status}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-3 sm:p-6 pt-0">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <Monitor className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> Device Name
              </p>
              <p className="text-xs sm:text-sm font-medium truncate">{agent.name || 'Unknown'}</p>
            </div>
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <Server className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> Hostname
              </p>
              <p className="text-xs sm:text-sm font-medium truncate">{agent.hostname || 'Unknown'}</p>
            </div>
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <Network className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> IP Address
              </p>
              <p className="text-xs sm:text-sm font-medium truncate">{agent.ipAddress || 'Unknown'}</p>
            </div>
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <Info className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> Type
              </p>
              <p className="text-xs sm:text-sm font-medium capitalize">{agent.type || 'Unknown'}</p>
            </div>
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <HardDrive className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> OS
              </p>
              <p className="text-xs sm:text-sm font-medium truncate">{agent.operatingSystem || agent.os || 'Unknown'}</p>
            </div>
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <Activity className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> Version
              </p>
              <p className="text-xs sm:text-sm font-medium">{agent.version || 'Unknown'}</p>
            </div>
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <Clock className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> Last Connected
              </p>
              <p className="text-xs sm:text-sm font-medium truncate">
                {agent.lastConnected || agent.lastHeartbeat 
                  ? new Date(agent.lastConnected || agent.lastHeartbeat!).toLocaleString()
                  : 'Never'}
              </p>
            </div>
            <div className="space-y-0.5 sm:space-y-1">
              <p className="text-[10px] sm:text-xs text-muted-foreground flex items-center gap-0.5 sm:gap-1">
                <Calendar className="h-2.5 w-2.5 sm:h-3 sm:w-3" /> Install Date
              </p>
              <p className="text-xs sm:text-sm font-medium">
                {agent.installDate || agent.createdAt
                  ? new Date(agent.installDate || agent.createdAt!).toLocaleDateString()
                  : 'Unknown'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <div className="space-y-3 sm:space-y-4">
        <h2 className="text-base sm:text-lg font-semibold">Performance Metrics</h2>
        
        <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-3 gap-2 sm:gap-4">
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
    </div>
  )
} 