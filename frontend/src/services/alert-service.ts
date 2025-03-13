import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { AlertFilters } from '../types/alert';

export function useAlerts(filters: AlertFilters) {
  return useQuery({
    queryKey: ['alerts', filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, value.toString());
      });
      const response = await fetch(`/alerts?${params}`);
      return response.json();
    }
  });
}

export function useUpdateAlertStatus() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ alertId, status }: { alertId: string; status: string }) => {
      const response = await fetch(`/alerts/${alertId}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    }
  });
}

export function useAssignAlert() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ alertId, userId }: { alertId: string; userId: string }) => {
      const response = await fetch(`/alerts/${alertId}/assign`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId })
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    }
  });
}

export function useAlertDetails(alertId: string) {
  return useQuery({
    queryKey: ['alert', alertId],
    queryFn: async () => {
      const response = await fetch(`/alerts/${alertId}`);
      return response.json();
    }
  });
}