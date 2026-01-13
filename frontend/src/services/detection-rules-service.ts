import { api } from '@/lib/api'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

// Types
export interface SigmaRule {
  id: string
  title: string
  description: string
  status: 'active' | 'disabled' | 'testing'
  level: 'critical' | 'high' | 'medium' | 'low'
  logsource: string
  tags: string[]
  lastModified: string
  matches: number
  content: string
}

export interface YaraRule {
  id: string
  name: string
  description: string
  category: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'active' | 'disabled' | 'testing'
  lastModified: string
  matches: number
  content: string
}

export interface RuleTestResult {
  ruleId: string
  success: boolean
  matches: number
  executionTime: number
  testedAt: string
}

// API functions
export const detectionRulesService = {
  // SIGMA Rules
  async getSigmaRules(params?: { status?: string; level?: string }): Promise<SigmaRule[]> {
    const queryParams = new URLSearchParams()
    if (params?.status) queryParams.append('status', params.status)
    if (params?.level) queryParams.append('level', params.level)
    
    const url = `/api/detection-rules/sigma${queryParams.toString() ? `?${queryParams}` : ''}`
    const response = await api.get<SigmaRule[]>(url)
    return response.data
  },

  async getSigmaRule(id: string): Promise<SigmaRule> {
    const response = await api.get<SigmaRule>(`/api/detection-rules/sigma/${id}`)
    return response.data
  },

  async createSigmaRule(rule: Partial<SigmaRule>): Promise<SigmaRule> {
    const response = await api.post<SigmaRule>('/api/detection-rules/sigma', rule)
    return response.data
  },

  async updateSigmaRule(id: string, rule: Partial<SigmaRule>): Promise<SigmaRule> {
    const response = await api.put<SigmaRule>(`/api/detection-rules/sigma/${id}`, rule)
    return response.data
  },

  async deleteSigmaRule(id: string): Promise<void> {
    await api.delete(`/api/detection-rules/sigma/${id}`)
  },

  async testSigmaRule(id: string): Promise<RuleTestResult> {
    const response = await api.post<RuleTestResult>(`/api/detection-rules/sigma/${id}/test`)
    return response.data
  },

  // YARA Rules
  async getYaraRules(params?: { status?: string; severity?: string }): Promise<YaraRule[]> {
    const queryParams = new URLSearchParams()
    if (params?.status) queryParams.append('status', params.status)
    if (params?.severity) queryParams.append('severity', params.severity)
    
    const url = `/api/detection-rules/yara${queryParams.toString() ? `?${queryParams}` : ''}`
    const response = await api.get<YaraRule[]>(url)
    return response.data
  },

  async getYaraRule(id: string): Promise<YaraRule> {
    const response = await api.get<YaraRule>(`/api/detection-rules/yara/${id}`)
    return response.data
  },

  async createYaraRule(rule: Partial<YaraRule>): Promise<YaraRule> {
    const response = await api.post<YaraRule>('/api/detection-rules/yara', rule)
    return response.data
  },

  async updateYaraRule(id: string, rule: Partial<YaraRule>): Promise<YaraRule> {
    const response = await api.put<YaraRule>(`/api/detection-rules/yara/${id}`, rule)
    return response.data
  },

  async deleteYaraRule(id: string): Promise<void> {
    await api.delete(`/api/detection-rules/yara/${id}`)
  },

  async testYaraRule(id: string): Promise<RuleTestResult> {
    const response = await api.post<RuleTestResult>(`/api/detection-rules/yara/${id}/test`)
    return response.data
  }
}

// React Query Hooks
export function useSigmaRules(params?: { status?: string; level?: string }) {
  return useQuery({
    queryKey: ['sigma-rules', params],
    queryFn: () => detectionRulesService.getSigmaRules(params),
    staleTime: 30000,
  })
}

export function useYaraRules(params?: { status?: string; severity?: string }) {
  return useQuery({
    queryKey: ['yara-rules', params],
    queryFn: () => detectionRulesService.getYaraRules(params),
    staleTime: 30000,
  })
}

export function useCreateSigmaRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (rule: Partial<SigmaRule>) => detectionRulesService.createSigmaRule(rule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sigma-rules'] })
    }
  })
}

export function useCreateYaraRule() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (rule: Partial<YaraRule>) => detectionRulesService.createYaraRule(rule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['yara-rules'] })
    }
  })
}

export function useTestSigmaRule() {
  return useMutation({
    mutationFn: (id: string) => detectionRulesService.testSigmaRule(id)
  })
}

export function useTestYaraRule() {
  return useMutation({
    mutationFn: (id: string) => detectionRulesService.testYaraRule(id)
  })
}
