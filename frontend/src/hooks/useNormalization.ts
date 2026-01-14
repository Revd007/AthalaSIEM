import { useQuery } from '@tanstack/react-query'
import { api, endpoints } from '@/lib/api'

export interface NormalizationStatistics {
  totalLogs: number
  normalizedLogs: number
  normalizationRate: number
  eventTypeDistribution: Array<{ eventType: string; count: number }>
  severityDistribution: Array<{ severity: number | null; count: number }>
  timeRange: {
    start: string
    end: string
  }
}

export interface NormalizedLog {
  id: string
  logEntryId: string
  timestamp: string
  sourceIp: string | null
  destinationIp: string | null
  eventType: string | null
  eventAction: string | null
  eventCategory: string | null
  severity: number | null
  userName: string | null
  processName: string | null
  processId: number | null
  protocol: string | null
  agentId: string | null
  hostName: string | null
}

export interface NormalizedLogsResponse {
  items: NormalizedLog[]
  totalCount: number
  page: number
  pageSize: number
  totalPages: number
}

export interface NormalizationFilters {
  page?: number
  pageSize?: number
  eventType?: string
  sourceIp?: string
  minSeverity?: number
  startDate?: string
  endDate?: string
}

export const useNormalizationStatistics = (startDate?: string, endDate?: string) => {
  return useQuery<NormalizationStatistics>({
    queryKey: ['normalization', 'statistics', startDate, endDate],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (startDate) params.append('startDate', startDate)
      if (endDate) params.append('endDate', endDate)
      
      const response = await api.get<NormalizationStatistics>(
        `${endpoints.normalization.statistics}?${params.toString()}`
      )
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })
}

export const useNormalizedLogs = (filters: NormalizationFilters = {}) => {
  return useQuery<NormalizedLogsResponse>({
    queryKey: ['normalization', 'normalized', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filters.page) params.append('page', filters.page.toString())
      if (filters.pageSize) params.append('pageSize', filters.pageSize.toString())
      if (filters.eventType) params.append('eventType', filters.eventType)
      if (filters.sourceIp) params.append('sourceIp', filters.sourceIp)
      if (filters.minSeverity) params.append('minSeverity', filters.minSeverity.toString())
      if (filters.startDate) params.append('startDate', filters.startDate)
      if (filters.endDate) params.append('endDate', filters.endDate)
      
      const response = await api.get<NormalizedLogsResponse>(
        `${endpoints.normalization.normalized}?${params.toString()}`
      )
      return response.data
    },
    refetchInterval: 10000, // Refresh every 10 seconds
  })
}
