import { api } from '@/lib/api'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

// Types
export interface AutomatedAction {
  id: string
  type: 'block' | 'isolate' | 'scan' | 'alert'
  trigger: string
  status: 'success' | 'failed' | 'in-progress'
  timestamp: string
  target: string
  details: string
  result?: string
}

export interface AutomatedRule {
  id: string
  name: string
  description: string
  status: 'active' | 'disabled'
  triggers: number
  lastTriggered?: string
  actionType: string
  conditions?: Record<string, any>
}

export interface AutomatedResponseStats {
  actionsToday: number
  successRate: number
  averageResponseTime: number
  activeRules: number
  totalActions: number
}

export interface ResponseMetric {
  time: string
  actions: number
  responseTime: number
}

export interface PaginatedResult<T> {
  items: T[]
  totalCount: number
  page: number
  pageSize: number
}

// API functions
export const automatedResponseService = {
  async getActions(params?: {
    type?: string
    status?: string
    page?: number
    pageSize?: number
  }): Promise<PaginatedResult<AutomatedAction>> {
    const queryParams = new URLSearchParams()
    if (params?.type) queryParams.append('type', params.type)
    if (params?.status) queryParams.append('status', params.status)
    if (params?.page) queryParams.append('page', params.page.toString())
    if (params?.pageSize) queryParams.append('pageSize', params.pageSize.toString())
    
    const url = `/api/automated-response/actions${queryParams.toString() ? `?${queryParams}` : ''}`
    const response = await api.get<PaginatedResult<AutomatedAction>>(url)
    return response.data
  },

  async getAction(id: string): Promise<AutomatedAction> {
    const response = await api.get<AutomatedAction>(`/api/automated-response/actions/${id}`)
    return response.data
  },

  async getRules(status?: string): Promise<AutomatedRule[]> {
    const url = status
      ? `/api/automated-response/rules?status=${status}`
      : '/api/automated-response/rules'
    const response = await api.get<AutomatedRule[]>(url)
    return response.data
  },

  async getRule(id: string): Promise<AutomatedRule> {
    const response = await api.get<AutomatedRule>(`/api/automated-response/rules/${id}`)
    return response.data
  },

  async createRule(rule: Partial<AutomatedRule>): Promise<AutomatedRule> {
    const response = await api.post<AutomatedRule>('/api/automated-response/rules', rule)
    return response.data
  },

  async updateRule(id: string, rule: Partial<AutomatedRule>): Promise<AutomatedRule> {
    const response = await api.put<AutomatedRule>(`/api/automated-response/rules/${id}`, rule)
    return response.data
  },

  async updateRuleStatus(id: string, status: string): Promise<AutomatedRule> {
    const response = await api.patch<AutomatedRule>(`/api/automated-response/rules/${id}/status`, { status })
    return response.data
  },

  async deleteRule(id: string): Promise<void> {
    await api.delete(`/api/automated-response/rules/${id}`)
  },

  async getStatistics(): Promise<AutomatedResponseStats> {
    const response = await api.get<AutomatedResponseStats>('/api/automated-response/statistics')
    return response.data
  },

  async getMetrics(hours: number = 24): Promise<ResponseMetric[]> {
    const response = await api.get<ResponseMetric[]>(`/api/automated-response/metrics?hours=${hours}`)
    return response.data
  }
}

// React Query Hooks
export function useAutomatedActions(params?: {
  type?: string
  status?: string
  page?: number
  pageSize?: number
}) {
  return useQuery({
    queryKey: ['automated-actions', params],
    queryFn: () => automatedResponseService.getActions(params),
    staleTime: 10000,
  })
}

export function useAutomatedRules(status?: string) {
  return useQuery({
    queryKey: ['automated-rules', status],
    queryFn: () => automatedResponseService.getRules(status),
    staleTime: 30000,
  })
}

export function useAutomatedResponseStats() {
  return useQuery({
    queryKey: ['automated-response-stats'],
    queryFn: () => automatedResponseService.getStatistics(),
    staleTime: 30000,
  })
}

export function useAutomatedResponseMetrics(hours: number = 24) {
  return useQuery({
    queryKey: ['automated-response-metrics', hours],
    queryFn: () => automatedResponseService.getMetrics(hours),
    staleTime: 30000,
  })
}

export function useCreateAutomatedRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (rule: Partial<AutomatedRule>) => automatedResponseService.createRule(rule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automated-rules'] })
    }
  })
}

export function useToggleRuleStatus() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => 
      automatedResponseService.updateRuleStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automated-rules'] })
    }
  })
}
