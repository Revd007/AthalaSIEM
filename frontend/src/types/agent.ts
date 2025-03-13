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

export interface Agent {
  agentId: string;
  name: string;
  hostname: string;
  ipAddress: string;
  port: number;
  os: string;
  type: string;
  status: AgentStatus;
  lastHeartbeat: string;
  createdAt: string;
  createdById?: string;
  apiKey?: string;
  isEnabled: boolean;
  configuration?: Record<string, string>;
  cpuUsage?: number;
  memoryUsage?: number;
  diskUsage?: number;
  collectEventLogs?: boolean;
  collectSystemMetrics?: boolean;
  eventLogsToMonitor?: string;
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