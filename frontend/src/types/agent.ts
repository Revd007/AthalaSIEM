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
  id: string; // Changed from agentId to match backend AgentDto.Id
  name: string;
  hostname: string;
  ipAddress: string; // Backend uses IpAddress (camelCase in JSON)
  port?: number; // Not in backend DTO, kept for compatibility
  os: string; // Maps to OperatingSystem in backend
  operatingSystem?: string; // Backend field name
  type: string;
  status: AgentStatus;
  lastHeartbeat?: string; // Maps to LastConnected in backend
  lastConnected?: string; // Backend field name
  createdAt?: string; // Maps to InstallDate in backend
  installDate?: string; // Backend field name
  createdById?: string; // Not in backend DTO
  apiKey?: string; // Not in backend DTO (security)
  isEnabled: boolean;
  enabled?: boolean; // Backend field name
  configuration?: Record<string, string>;
  cpuUsage?: number;
  memoryUsage?: number;
  diskUsage?: number;
  collectEventLogs?: boolean;
  collectSystemMetrics?: boolean;
  eventLogsToMonitor?: string[]; // Backend uses List<string>
  version?: string;
  healthStatus?: string; // Backend field
  tags?: string[]; // Backend field
  osInfo?: {
    platform: string;
    version: string;
    architecture: string;
  };
  cloudInfo?: {
    provider: string;
    region: string;
    instanceId: string;
  };
  metrics?: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkIn?: number;
    networkOut?: number;
    processCount?: number;
    threadCount?: number;
  };
  alerts?: {
    count: number;
    severity: Severity;
    lastAlert?: string;
  };
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
  category?: string;
  properties?: Record<string, any>;
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

export interface AgentMetrics {
  timestamp: string;
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkIn?: number;
  networkOut?: number;
  processCount?: number;
  threadCount?: number;
}

export interface AgentAlert {
  id: string;
  agentId: string;
  timestamp: string;
  type: string;
  message: string;
  severity: Severity;
  status: 'open' | 'resolved' | 'acknowledged';
  metadata?: Record<string, any>;
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