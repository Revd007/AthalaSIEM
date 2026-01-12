import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { AlertFilters, AlertQueryParams, PaginatedResult } from '../types/alert';
import { api } from '@/lib/api';
import { Alert } from '@/types/alert';

export function useAlerts(filters: AlertFilters = {}) {
  return useQuery({
    queryKey: ['alerts', filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      
      // Map frontend filters to backend AlertQueryDto
      if (filters.search) params.append('searchTerm', filters.search);
      if (filters.severity) params.append('severity', filters.severity);
      if (filters.status) params.append('status', filters.status);
      if (filters.agentId) params.append('agentId', filters.agentId);
      if (filters.source) params.append('source', filters.source);
      if (filters.ruleId) params.append('ruleId', filters.ruleId);
      if (filters.assignedTo) params.append('assignedTo', filters.assignedTo);
      if (filters.startTime) params.append('startTime', filters.startTime);
      if (filters.endTime) params.append('endTime', filters.endTime);
      if (filters.limit) params.append('limit', filters.limit.toString());
      if (filters.offset) params.append('offset', filters.offset.toString());
      if (filters.sortField) params.append('sortField', filters.sortField);
      if (filters.sortDirection) params.append('sortDirection', filters.sortDirection);
      
      // Backend returns PaginatedResult<AlertDto>
      const { data } = await api.get<PaginatedResult<Alert>>(`/api/alerts?${params.toString()}`);
      
      // Return items array for backward compatibility, or full paginated result
      return data?.items ?? [];
    }
  });
}

export function useAlertsPaginated(filters: AlertFilters = {}) {
  return useQuery({
    queryKey: ['alerts-paginated', filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      
      if (filters.search) params.append('searchTerm', filters.search);
      if (filters.severity) params.append('severity', filters.severity);
      if (filters.status) params.append('status', filters.status);
      if (filters.agentId) params.append('agentId', filters.agentId);
      if (filters.source) params.append('source', filters.source);
      if (filters.ruleId) params.append('ruleId', filters.ruleId);
      if (filters.assignedTo) params.append('assignedTo', filters.assignedTo);
      if (filters.startTime) params.append('startTime', filters.startTime);
      if (filters.endTime) params.append('endTime', filters.endTime);
      if (filters.limit) params.append('limit', filters.limit.toString());
      if (filters.offset) params.append('offset', filters.offset.toString());
      if (filters.sortField) params.append('sortField', filters.sortField);
      if (filters.sortDirection) params.append('sortDirection', filters.sortDirection);
      
      const { data } = await api.get<PaginatedResult<Alert>>(`/api/alerts?${params.toString()}`);
      return data;
    }
  });
}

export function useUpdateAlertStatus() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ 
      alertId, 
      status, 
      assignedTo, 
      comment, 
      closeReason, 
      updatedBy 
    }: { 
      alertId: string; 
      status: string;
      assignedTo?: string;
      comment?: string;
      closeReason?: string;
      updatedBy?: string;
    }) => {
      const { data } = await api.patch<Alert>(`/api/alerts/${alertId}/status`, {
        status,
        assignedTo: assignedTo || '',
        comment: comment || '',
        closeReason: closeReason || '',
        updatedBy: updatedBy || '',
        updatedAt: new Date().toISOString()
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
      queryClient.invalidateQueries({ queryKey: ['alerts-paginated'] });
    }
  });
}

export function useAssignAlert() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ alertId, userId }: { alertId: string; userId: string }) => {
      const { data } = await api.patch<Alert>(`/api/alerts/${alertId}/assign`, {
        assignedTo: userId,
        assignedToUserId: userId,
        updatedBy: userId,
        updatedAt: new Date().toISOString()
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
      queryClient.invalidateQueries({ queryKey: ['alerts-paginated'] });
    }
  });
}

export function useAlertDetails(alertId: string) {
  return useQuery({
    queryKey: ['alert', alertId],
    queryFn: async () => {
      const { data } = await api.get<Alert>(`/api/alerts/${alertId}`);
      return data;
    }
  });
}

export function useAddAlertComment() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ 
      alertId, 
      comment, 
      author 
    }: { 
      alertId: string; 
      comment: string; 
      author: string;
    }) => {
      const { data } = await api.post<Alert>(`/api/alerts/${alertId}/comments`, {
        comment,
        author,
        createdAt: new Date().toISOString()
      });
      return data;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['alert', variables.alertId] });
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    }
  });
}