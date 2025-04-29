export enum AgentStatus {
  Pending = 'Pending',
  Online = 'Online',
  Offline = 'Offline',
  Error = 'Error',
  Active = 'Active'
}

export type AgentType = 'windows' | 'linux' | 'cloud';
export type CloudProvider = 'aws' | 'azure' | 'gcp';

export interface NewAgentConfig {
  name: string;
  hostname: string;
  ipAddress: string;
  port?: number;
  os: string;
  type?: string;
}

export interface AgentConfig {
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
  logs: Array<{
    timestamp: string
    level: string
    message: string
  }>
  alerts: Array<{
    id: string
    severity: string
    message: string
    timestamp: string
  }>
}

export interface DeploymentToken {
  token: string
  expiresAt: string
  downloadUrl: string
}

export interface Agent {
  id: string
  name: string
  type: string
  status: string
  lastSeen: string
  version: string
  os: string
  ip: string
  location: string
  hostname: string
  ipAddress: string
  port?: number
  isEnabled: boolean
  collectEventLogs: boolean
  collectSystemMetrics: boolean
  eventLogsToMonitor?: string
  configuration?: Record<string, string>
  resources: {
    cpu: number
    memory: number
    disk: number
  }
  agentId?: string // For backward compatibility
}

export interface AgentLog {
  id: string
  timestamp: string
  level: string
  message: string
  source: string
}

export interface AgentAlert {
  id: string
  timestamp: string
  severity: string
  message: string
  source: string
  status: string
}

export interface AgentMetrics {
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

export enum HealthStatus {
  Healthy = 'Healthy',
  Warning = 'Warning',
  Critical = 'Critical'
}

export interface HealthMetric {
  name: string;
  value: number;
  unit: string;
  status: HealthStatus;
}

export interface AgentHealthReport {
  agentId: string;
  timestamp: string;
  overallStatus: string;
  metrics: Record<string, HealthMetric>;
}

export enum Severity {
  Low = 'Low',
  Medium = 'Medium',
  High = 'High',
  Critical = 'Critical'
}

export interface LogEntry {
  id: string;
  agentId: string;
  timestamp: string;
  source: string;
  level: string;
  message: string;
  eventId?: number;
  processId?: number;
  processName?: string;
  threadId?: number;
  machineName?: string;
  ipAddress?: string;
  username?: string;
  severity: Severity;
}

export interface LogQueryParams {
  startDate?: Date;
  endDate?: Date;
  severity?: string;
  source?: string;
  page?: number;
  pageSize?: number;
}

export interface PaginatedResult<T> {
  items: T[];
  totalCount: number;
  pageCount: number;
  currentPage: number;
  pageSize: number;
}

// "id": "48865b6d-944e-40cf-97fa-658315c5804d",
//         "name": "DESKTOP-2F0SABS",
//         "hostname": "DESKTOP-2F0SABS",
//         "ipAddress": "192.168.13.7",
//         "os": "Windows",
//         "status": "Pending",
//         "lastHeartbeat": "2025-02-20T06:33:37",
//         "enabled": true,
//         "eventCount": 0