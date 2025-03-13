import { useQuery } from '@tanstack/react-query'
import type { 
  AIStatusResponse, 
  SecurityEventResponse,
  SystemMetricsResponse 
} from '../types/api'

export function useStats() {
  return useQuery<AIStatusResponse>({
    queryKey: ['stats'],
    queryFn: async () => {
      const response = await fetch('/api/stats')
      return response.json()
    }
  })
}

export function useRecentAlerts() {
  return useQuery({
    queryKey: ['recent-alerts'],
    queryFn: async () => {
      const response = await fetch('/api/recent-alerts')
      return response.json()
    },
    refetchInterval: 30000
  })
}

export function useSystemHealth() {
  return useQuery({
    queryKey: ['system-health'],
    queryFn: async () => {
      const response = await fetch('/api/system-health')
      return response.json()
    },
    refetchInterval: 15000
  })
}

export function useSystemMetrics() {
  return useQuery({
    queryKey: ['system-metrics'],
    queryFn: async () => {
      const response = await fetch('/api/metrics')
      return response.json()
    },
    refetchInterval: 30000
  })
}
export function useEventsAnalysis() {
  return useQuery({
    queryKey: ['events-analysis'],
    queryFn: async () => {
      const response = await fetch('/api/events-analysis')
      return response.json()
    }
  })
}

export function useKnowledgeGraph() {
  return useQuery({
    queryKey: ['knowledge-graph'],
    queryFn: async () => {
      const response = await fetch('/api/knowledge-graph')
      return response.json()
    }
  })
}
