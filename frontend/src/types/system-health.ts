export type DeviceType = 'server' | 'firewall' | 'network' | 'endpoint' | 'storage' | 'security-appliance'
export type DeviceStatus = 'healthy' | 'warning' | 'critical' | 'offline'
export type MetricType = 'cpu' | 'memory' | 'disk' | 'network' | 'temperature' | 'power' | 'updates'

export interface SystemDevice {
  id: string
  name: string
  type: DeviceType
  status: DeviceStatus
  ipAddress: string
  location: string
  lastSeen: string
  agentVersion: string
  operatingSystem?: string
  manufacturer?: string
  model?: string
}

export interface DeviceMetric {
  id: string
  deviceId: string
  type: MetricType
  value: number
  unit: string
  timestamp: string
  threshold: {
    warning: number
    critical: number
  }
  history: Array<{
    value: number
    timestamp: string
  }>
}

export interface DeviceHealth {
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
  processes: Array<{
    pid: number
    name: string
    cpu: number
    memory: number
    status: string
  }>
  services: Array<{
    name: string
    status: 'running' | 'stopped' | 'error'
    uptime: number
  }>
}

export interface Device {
  id: string
  name: string
  type: string
  status: string
  lastSeen: string
  version: string
  os: string
  ip: string
  location: string
  resources: {
    cpu: number
    memory: number
    disk: number
  }
}

export interface SystemMetrics {
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
}

export interface SystemAlert {
  id: string
  timestamp: string
  severity: string
  message: string
  source: string
  status: string
}

export interface SystemLog {
  id: string
  timestamp: string
  level: string
  message: string
  source: string
} 