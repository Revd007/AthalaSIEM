import { apiService } from './api-service'

export interface SecurityAlert {
  id: string
  title: string
  description: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'new' | 'investigating' | 'resolved' | 'closed'
  source: string
  timestamp: string
  affectedAssets: string[]
  assignedTo?: string
  tags: string[]
  metadata: Record<string, any>
}

export interface AlertFilter {
  startDate?: string
  endDate?: string
  severity?: string[]
  status?: string[]
  source?: string[]
  assignedTo?: string
  search?: string
  page?: number
  limit?: number
}

export class SecurityAlertService {
  static async getAlerts(filters: AlertFilter) {
    return await apiService.get<{
      data: SecurityAlert[]
      total: number
      page: number
      limit: number
    }>('/security-alerts', { params: filters })
  }

  static async getAlertById(id: string) {
    return await apiService.get<SecurityAlert>(`/security-alerts/${id}`)
  }

  static async updateAlert(id: string, data: Partial<SecurityAlert>) {
    return await apiService.put<SecurityAlert>(`/security-alerts/${id}`, data)
  }

  static async bulkUpdateAlerts(ids: string[], data: Partial<SecurityAlert>) {
    return await apiService.post<{ success: boolean }>(
      '/security-alerts/bulk-update',
      { ids, ...data }
    )
  }

  static async getAlertStats() {
    return await apiService.get<{
      totalAlerts: number
      alertsBySeverity: Record<string, number>
      alertsByStatus: Record<string, number>
      alertsBySource: Record<string, number>
      trendData: {
        date: string
        count: number
        severity: string
      }[]
    }>('/security-alerts/stats')
  }
}