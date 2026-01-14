import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, endpoints } from '@/lib/api'
import { toast } from 'sonner'

export interface CorrelationStatistics {
  totalCorrelationAlerts: number
  ruleBreakdown: Array<{ ruleName: string; count: number }>
  severityBreakdown: Array<{ severity: string; count: number }>
  timeRange: {
    start: string
    end: string
  }
}

export interface CorrelationRule {
  name: string
  description: string
  threshold: number
  timeWindow: number
}

export interface CorrelationResult {
  logEntryId: string
  correlationResults: Array<{
    ruleName: string | null
    ruleDescription: string | null
    correlationId: string
    type: string
    confidence: number
    correlatedLogCount: number
    metadata: Record<string, any>
  }>
}

export const useCorrelationStatistics = (startDate?: string, endDate?: string) => {
  return useQuery<CorrelationStatistics>({
    queryKey: ['correlation', 'statistics', startDate, endDate],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (startDate) params.append('startDate', startDate)
      if (endDate) params.append('endDate', endDate)
      
      const response = await api.get<CorrelationStatistics>(
        `${endpoints.correlation.statistics}?${params.toString()}`
      )
      return response.data
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  })
}

export const useCorrelationRules = () => {
  return useQuery<CorrelationRule[]>({
    queryKey: ['correlation', 'rules'],
    queryFn: async () => {
      const response = await api.get<CorrelationRule[]>(endpoints.correlation.rules)
      return response.data
    },
  })
}

export const useTriggerCorrelation = () => {
  const queryClient = useQueryClient()
  
  return useMutation<CorrelationResult, Error, string>({
    mutationFn: async (logEntryId: string) => {
      const response = await api.post<CorrelationResult>(
        endpoints.correlation.trigger(logEntryId)
      )
      return response.data
    },
    onSuccess: (data) => {
      toast.success(`Correlation triggered: ${data.correlationResults.length} results found`)
      queryClient.invalidateQueries({ queryKey: ['correlation'] })
    },
    onError: (error) => {
      toast.error(`Failed to trigger correlation: ${error.message}`)
    },
  })
}
