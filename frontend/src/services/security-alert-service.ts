import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

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

export function useAlerts(filters: AlertFilter) {
  return useQuery({
    queryKey: ['alerts', filters],
    queryFn: async () => {
      const params = new URLSearchParams()
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, value.toString())
      })
      const response = await fetch(`/security-alerts?${params}`)
      return response.json()
    }
  })
}

export function useAlert(id: string) {
  return useQuery({
    queryKey: ['alert', id],
    queryFn: async () => {
      const response = await fetch(`/security-alerts/${id}`)
      return response.json()
    }
  })
}

export function useUpdateAlert() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async ({id, data}: {id: string, data: Partial<SecurityAlert>}) => {
      const response = await fetch(`/security-alerts/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      })
      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] })
    }
  })
}

export function useBulkUpdateAlerts() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async ({ids, data}: {ids: string[], data: Partial<SecurityAlert>}) => {
      const response = await fetch('/security-alerts/bulk-update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids, ...data })
      })
      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] })
    }
  })
}

export function useAlertStats() {
  return useQuery({
    queryKey: ['alert-stats'],
    queryFn: async () => {
      const response = await fetch('/security-alerts/stats')
      return response.json()
    },
    refetchInterval: 30000
  })
}