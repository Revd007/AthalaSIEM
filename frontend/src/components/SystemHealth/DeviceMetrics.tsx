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

interface DeviceMetrics {
  cpu: {
    usage: number
    temperature: number
    cores: number
  }
  memory: {
    total: number
    used: number
    free: number
    swap: {
      total: number
      used: number
      free: number
    }
  }
  disk: Array<{
    path: string
    total: number
    used: number
    free: number
    mountPoint: string
  }>
  network: Array<{
    interface: string
    bytesIn: number
    bytesOut: number
    packetsIn: number
    packetsOut: number
    errors: number
    drops: number
  }>
  processes: Array<unknown>
  services: Array<unknown>
}

interface DeviceMetricsProps {
  deviceId: string
  metrics: DeviceMetrics
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

// Mock data - replace with actual API call
const mockMetrics: DeviceHealth = {
  cpu: {
    usage: 45,
    temperature: 65,
    cores: 8
  },
  memory: {
    total: 32768,
    used: 16384,
    free: 16384,
    swap: {
      total: 8192,
      used: 1024,
      free: 7168
    }
  },
  disk: [
    {
      path: '/dev/sda1',
      total: 512000,
      used: 256000,
      free: 256000,
      mountPoint: '/'
    }
  ],
  network: [
    {
      interface: 'eth0',
      bytesIn: 1024000,
      bytesOut: 512000,
      packetsIn: 1000,
      packetsOut: 500,
      errors: 0,
      drops: 0
    }
  ],
  processes: [],
  services: []
}

function formatBytes(bytes: number): string {
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  if (bytes === 0) return '0 B'
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
}

export function DeviceMetrics({ deviceId, metrics }: DeviceMetricsProps) {
  const cpuUsagePercent = metrics.cpu.usage
  const memoryUsagePercent = (metrics.memory.used / metrics.memory.total) * 100
  const diskUsagePercent = (metrics.disk[0].used / metrics.disk[0].total) * 100

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Performance Metrics</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          title="CPU Usage"
          value={`${cpuUsagePercent}%`}
          icon={Cpu}
          secondaryValue={`${metrics.cpu.cores} Cores | ${metrics.cpu.temperature}°C`}
          progress={cpuUsagePercent}
        />
        
        <MetricCard
          title="Memory Usage"
          value={formatBytes(metrics.memory.used)}
          icon={CircuitBoard}
          secondaryValue={`Total: ${formatBytes(metrics.memory.total)}`}
          progress={memoryUsagePercent}
        />
        
        <MetricCard
          title="Disk Usage"
          value={formatBytes(metrics.disk[0].used)}
          icon={HardDrive}
          secondaryValue={`Free: ${formatBytes(metrics.disk[0].free)}`}
          progress={diskUsagePercent}
        />
        
        <MetricCard
          title="Network In"
          value={formatBytes(metrics.network[0].bytesIn)}
          icon={Network}
          secondaryValue={`${metrics.network[0].packetsIn} packets`}
        />
        
        <MetricCard
          title="Network Out"
          value={formatBytes(metrics.network[0].bytesOut)}
          icon={Activity}
          secondaryValue={`${metrics.network[0].packetsOut} packets`}
        />
        
        <MetricCard
          title="Temperature"
          value={`${metrics.cpu.temperature}°C`}
          icon={Thermometer}
          progress={metrics.cpu.temperature}
        />
      </div>
    </div>
  )
} 