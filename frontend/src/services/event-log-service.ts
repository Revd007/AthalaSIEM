import { apiService } from './api-service'

export interface EventLog {
  id: string
  timestamp: string
  eventType: string
  source: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  rawData: any
  metadata: Record<string, any>
}

export interface EventLogFilter {
  startDate?: string
  endDate?: string
  eventType?: string[]
  source?: string[]
  severity?: string[]
  search?: string
  page?: number
  limit?: number
}

export class EventLogService {
  static async getEventLogs(filters: EventLogFilter) {
    return await apiService.get<{
      data: EventLog[]
      total: number
      page: number
      limit: number
    }>('/event-logs', { params: filters })
  }

  static async getEventLogById(id: string) {
    return await apiService.get<EventLog>(`/event-logs/${id}`)
  }

  static async getEventLogStats() {
    return await apiService.get<{
      totalEvents: number
      eventsByType: Record<string, number>
      eventsBySeverity: Record<string, number>
      eventsBySource: Record<string, number>
    }>('/event-logs/stats')
  }

  static async exportEventLogs(filters: EventLogFilter, format: 'csv' | 'json') {
    return await apiService.get('/event-logs/export', {
      params: { ...filters, format },
      responseType: 'blob',
    })
  }
}