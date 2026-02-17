import { api } from '@/lib/api';
import type { LogEntry, PaginatedResult } from '@/types/agent';

export interface LogQueryParams {
  agentId?: string;
  startDate?: string;
  endDate?: string;
  severity?: string;
  source?: string;
  searchTerm?: string;
  limit?: number;
  offset?: number;
  sortField?: string;
  sortDirection?: 'asc' | 'desc';
}

export interface LogEntryDto {
  id: string;
  agentId?: string;
  timestamp: string;
  source: string;
  level: string;
  message: string;
  severity: string;
  eventId?: number;
  processId?: number;
  processName?: string;
  threadId?: number;
  machineName?: string;
  ipAddress?: string;
  username?: string;
  rawLog?: string;
  category?: string;
  properties?: Record<string, any>;
}

export const logService = {
  async getLogs(params: LogQueryParams = {}): Promise<PaginatedResult<LogEntry>> {
    const queryString = new URLSearchParams();
    
    if (params.agentId) queryString.append('agentId', params.agentId);
    if (params.startDate) queryString.append('startTime', params.startDate);
    if (params.endDate) queryString.append('endTime', params.endDate);
    if (params.severity) queryString.append('severity', params.severity);
    if (params.source) queryString.append('source', params.source);
    if (params.searchTerm) queryString.append('searchTerm', params.searchTerm);
    if (params.limit) queryString.append('limit', params.limit.toString());
    if (params.offset) queryString.append('offset', params.offset.toString());
    if (params.sortField) queryString.append('sortField', params.sortField);
    if (params.sortDirection) queryString.append('sortDirection', params.sortDirection);
    
    const { data } = await api.get<PaginatedResult<LogEntryDto>>(`/api/logs?${queryString.toString()}`);
    
    // Map LogEntryDto to LogEntry
    const mappedItems = (data?.items ?? []).map((dto): LogEntry => {
      // Convert severity string to Severity enum
      let severity: any = 'Low';
      if (dto.severity) {
        const sevLower = dto.severity.toLowerCase();
        if (sevLower === 'critical') severity = 'Critical';
        else if (sevLower === 'high') severity = 'High';
        else if (sevLower === 'medium') severity = 'Medium';
        else severity = 'Low';
      } else if (dto.level) {
        const levelLower = dto.level.toLowerCase();
        if (levelLower === 'critical' || levelLower === 'fatal') severity = 'Critical';
        else if (levelLower === 'error') severity = 'High';
        else if (levelLower === 'warning') severity = 'Medium';
        else severity = 'Low';
      }
      
      return {
        id: dto.id,
        agentId: dto.agentId || '',
        timestamp: typeof dto.timestamp === 'string' ? dto.timestamp : new Date(dto.timestamp).toISOString(),
        source: dto.source || 'Unknown',
        level: dto.level || 'Information',
        message: dto.message || '',
        severity: severity,
        eventId: dto.eventId,
        processId: dto.processId,
        processName: dto.processName,
        threadId: dto.threadId,
        machineName: dto.machineName || (dto as any).computerName || '',
        ipAddress: dto.ipAddress || (dto as any).clientIp || '',
        username: dto.username || (dto as any).userId || '',
        category: dto.category || '',
        properties: dto.properties || {},
      };
    });
    
    return {
      items: mappedItems,
      totalCount: data?.totalCount ?? 0,
      pageCount: data?.pageCount ?? 0,
      currentPage: data?.currentPage ?? 1,
      pageSize: data?.pageSize ?? 100,
    };
  },

  async getLogById(id: string): Promise<LogEntry | null> {
    try {
      const { data } = await api.get<LogEntryDto>(`/api/logs/${id}`);
      if (!data) return null;
      
      // Convert severity
      let severity: any = 'Low';
      if (data.severity) {
        const sevLower = data.severity.toLowerCase();
        if (sevLower === 'critical') severity = 'Critical';
        else if (sevLower === 'high') severity = 'High';
        else if (sevLower === 'medium') severity = 'Medium';
        else severity = 'Low';
      } else if (data.level) {
        const levelLower = data.level.toLowerCase();
        if (levelLower === 'critical' || levelLower === 'fatal') severity = 'Critical';
        else if (levelLower === 'error') severity = 'High';
        else if (levelLower === 'warning') severity = 'Medium';
        else severity = 'Low';
      }
      
      return {
        id: data.id,
        agentId: data.agentId || '',
        timestamp: typeof data.timestamp === 'string' ? data.timestamp : new Date(data.timestamp).toISOString(),
        source: data.source || data.category || 'Unknown',
        level: data.level || 'Information',
        message: data.message || '',
        severity: severity,
        eventId: data.eventId,
        processId: data.processId,
        processName: data.processName,
        threadId: data.threadId,
        machineName: data.machineName || data.computerName || '',
        ipAddress: data.ipAddress || data.clientIp || '',
        username: data.username || data.userId || '',
      };
    } catch (error) {
      console.error('Error fetching log:', error);
      return null;
    }
  },

  async getLogsBySeverity(severity: string, limit: number = 100, offset: number = 0): Promise<PaginatedResult<LogEntry>> {
    const { data } = await api.get<PaginatedResult<LogEntryDto>>(`/api/logs/severity/${severity}?limit=${limit}&offset=${offset}`);
    
    const mappedItems = (data?.items ?? []).map((dto): LogEntry => {
      // Convert severity string to Severity enum
      let mappedSeverity: any = 'Low';
      if (dto.severity) {
        const sevLower = dto.severity.toLowerCase();
        if (sevLower === 'critical') mappedSeverity = 'Critical';
        else if (sevLower === 'high') mappedSeverity = 'High';
        else if (sevLower === 'medium') mappedSeverity = 'Medium';
        else mappedSeverity = 'Low';
      } else if (dto.level) {
        const levelLower = dto.level.toLowerCase();
        if (levelLower === 'critical' || levelLower === 'fatal') mappedSeverity = 'Critical';
        else if (levelLower === 'error') mappedSeverity = 'High';
        else if (levelLower === 'warning') mappedSeverity = 'Medium';
        else mappedSeverity = 'Low';
      }
      
      return {
        id: dto.id,
        agentId: dto.agentId || '',
        timestamp: typeof dto.timestamp === 'string' ? dto.timestamp : new Date(dto.timestamp).toISOString(),
        source: dto.source || dto.category || 'Unknown',
        level: dto.level || 'Information',
        message: dto.message || '',
        severity: mappedSeverity,
        eventId: dto.eventId,
        processId: dto.processId,
        processName: dto.processName,
        threadId: dto.threadId,
        machineName: dto.machineName || dto.computerName || '',
        ipAddress: dto.ipAddress || dto.clientIp || '',
        username: dto.username || dto.userId || '',
      };
    });
    
    return {
      items: mappedItems,
      totalCount: data?.totalCount ?? 0,
      pageCount: data?.pageCount ?? 0,
      currentPage: data?.currentPage ?? 1,
      pageSize: data?.pageSize ?? limit,
    };
  },

  async getLogsByTimeRange(start: Date, end: Date, limit: number = 100, offset: number = 0): Promise<PaginatedResult<LogEntry>> {
    const queryString = new URLSearchParams({
      start: start.toISOString(),
      end: end.toISOString(),
      limit: limit.toString(),
      offset: offset.toString()
    });
    
    const { data } = await api.get<PaginatedResult<LogEntryDto>>(`/api/logs/timerange?${queryString.toString()}`);
    
    const mappedItems = (data?.items ?? []).map((dto): LogEntry => {
      // Convert severity string to Severity enum
      let mappedSeverity: any = 'Low';
      if (dto.severity) {
        const sevLower = dto.severity.toLowerCase();
        if (sevLower === 'critical') mappedSeverity = 'Critical';
        else if (sevLower === 'high') mappedSeverity = 'High';
        else if (sevLower === 'medium') mappedSeverity = 'Medium';
        else mappedSeverity = 'Low';
      } else if (dto.level) {
        const levelLower = dto.level.toLowerCase();
        if (levelLower === 'critical' || levelLower === 'fatal') mappedSeverity = 'Critical';
        else if (levelLower === 'error') mappedSeverity = 'High';
        else if (levelLower === 'warning') mappedSeverity = 'Medium';
        else mappedSeverity = 'Low';
      }
      
      return {
        id: dto.id,
        agentId: dto.agentId || '',
        timestamp: typeof dto.timestamp === 'string' ? dto.timestamp : new Date(dto.timestamp).toISOString(),
        source: dto.source || dto.category || 'Unknown',
        level: dto.level || 'Information',
        message: dto.message || '',
        severity: mappedSeverity,
        eventId: dto.eventId,
        processId: dto.processId,
        processName: dto.processName,
        threadId: dto.threadId,
        machineName: dto.machineName || dto.computerName || '',
        ipAddress: dto.ipAddress || dto.clientIp || '',
        username: dto.username || dto.userId || '',
      };
    });
    
    return {
      items: mappedItems,
      totalCount: data?.totalCount ?? 0,
      pageCount: data?.pageCount ?? 0,
      currentPage: data?.currentPage ?? 1,
      pageSize: data?.pageSize ?? limit,
    };
  }
};
