import { api } from '@/lib/api'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

// Types
export interface ChangeRequest {
  id: string
  title: string
  type: 'emergency' | 'standard' | 'normal'
  status: 'pending' | 'approved' | 'rejected' | 'implemented'
  requester: string
  dateSubmitted: string
  implementation?: string
  risk: 'low' | 'medium' | 'high'
  approvers: string[]
  description: string
}

export interface ChangeManagementStats {
  totalRequests: number
  pendingRequests: number
  approvedRequests: number
  implementedRequests: number
  rejectedRequests: number
  emergencyRequests: number
  highRiskRequests: number
}

// API functions
export const changeManagementService = {
  async getChangeRequests(params?: {
    status?: string
    type?: string
    risk?: string
  }): Promise<ChangeRequest[]> {
    const queryParams = new URLSearchParams()
    if (params?.status) queryParams.append('status', params.status)
    if (params?.type) queryParams.append('type', params.type)
    if (params?.risk) queryParams.append('risk', params.risk)
    
    const url = `/api/change-management${queryParams.toString() ? `?${queryParams}` : ''}`
    const response = await api.get<ChangeRequest[]>(url)
    return response.data
  },

  async getChangeRequest(id: string): Promise<ChangeRequest> {
    const response = await api.get<ChangeRequest>(`/api/change-management/${id}`)
    return response.data
  },

  async createChangeRequest(request: Partial<ChangeRequest>): Promise<ChangeRequest> {
    const response = await api.post<ChangeRequest>('/api/change-management', request)
    return response.data
  },

  async updateChangeRequest(id: string, request: Partial<ChangeRequest>): Promise<ChangeRequest> {
    const response = await api.put<ChangeRequest>(`/api/change-management/${id}`, request)
    return response.data
  },

  async updateStatus(id: string, status: string): Promise<ChangeRequest> {
    const response = await api.patch<ChangeRequest>(`/api/change-management/${id}/status`, { status })
    return response.data
  },

  async deleteChangeRequest(id: string): Promise<void> {
    await api.delete(`/api/change-management/${id}`)
  },

  async getStatistics(): Promise<ChangeManagementStats> {
    const response = await api.get<ChangeManagementStats>('/api/change-management/statistics')
    return response.data
  }
}

// React Query Hooks
export function useChangeRequests(params?: {
  status?: string
  type?: string
  risk?: string
}) {
  return useQuery({
    queryKey: ['change-requests', params],
    queryFn: () => changeManagementService.getChangeRequests(params),
    staleTime: 30000,
  })
}

export function useChangeManagementStats() {
  return useQuery({
    queryKey: ['change-management-stats'],
    queryFn: () => changeManagementService.getStatistics(),
    staleTime: 60000,
  })
}

export function useCreateChangeRequest() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: Partial<ChangeRequest>) => changeManagementService.createChangeRequest(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['change-requests'] })
      queryClient.invalidateQueries({ queryKey: ['change-management-stats'] })
    }
  })
}

export function useUpdateChangeStatus() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => 
      changeManagementService.updateStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['change-requests'] })
      queryClient.invalidateQueries({ queryKey: ['change-management-stats'] })
    }
  })
}
