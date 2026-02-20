/**
 * File Integrity Monitoring (FIM) types matching .NET backend DTOs.
 * Backend: api/FileIntegrity (FileIntegrityController).
 */

export interface FIMEvent {
  id: string
  agentId: string
  agentName: string
  filePath: string
  changeType: string
  baselineHash?: string | null
  currentHash?: string | null
  baselineSize?: number | null
  currentSize?: number | null
  baselineModified?: string | null
  currentModified?: string | null
  fileAttributes?: string | null
  severity: string
  detectedAt: string
  processedAt: string
  isAcknowledged: boolean
  acknowledgedBy?: string | null
  acknowledgedAt?: string | null
  details?: string | null
}

export interface FIMRule {
  id: string
  name: string
  description?: string | null
  isEnabled: boolean
  monitoredPaths: string
  excludePatterns?: string | null
  realTimeMonitoring: boolean
  scanIntervalMinutes: number
  severity: string
  alertOnCreation: boolean
  alertOnModification: boolean
  alertOnDeletion: boolean
  alertOnRename: boolean
  createdAt: string
  updatedAt: string
  createdBy?: string | null
  targetAgents?: string | null
}

export interface FIMStats {
  totalEvents: number
  eventsBySeverity: Array<{ severity: string; count: number }>
  eventsByChangeType: Array<{ changeType: string; count: number }>
  eventsByAgent: Array<{ agentId: string; agentName: string; count: number }>
  acknowledgedEvents: number
  unacknowledgedEvents: number
  eventsOverTime: Array<{ date: string; count: number }>
}

export interface PagedFIMEvents {
  items: FIMEvent[]
  totalCount: number
  page: number
  pageSize: number
  totalPages: number
}

export interface FIMEventsQuery {
  agentId?: string
  severity?: string
  changeType?: string
  acknowledged?: boolean
  startDate?: string
  endDate?: string
  page?: number
  pageSize?: number
}

export interface CreateFIMRuleRequest {
  name: string
  description?: string
  isEnabled?: boolean
  monitoredPaths: string
  excludePatterns?: string
  realTimeMonitoring?: boolean
  scanIntervalMinutes?: number
  severity?: string
  alertOnCreation?: boolean
  alertOnModification?: boolean
  alertOnDeletion?: boolean
  alertOnRename?: boolean
  targetAgents?: string
}
