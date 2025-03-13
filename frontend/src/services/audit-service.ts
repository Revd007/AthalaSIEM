import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

interface AuditLog {
  id: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  details: any;
  ip_address: string;
  timestamp: string;
}

export function useAuditLogs(params: {
  startDate?: string;
  endDate?: string;
  userId?: string;
  action?: string;
  resourceType?: string;
}) {
  return useQuery({
    queryKey: ['audit-logs', params],
    queryFn: async () => {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        if (value) searchParams.append(key, value);
      });
      const response = await fetch(`/audit-logs?${searchParams}`);
      return response.json() as Promise<AuditLog[]>;
    }
  });
}

export function useCreateAuditEntry() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (data: Partial<AuditLog>) => {
      const response = await fetch('/audit-logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });
      return response.json() as Promise<AuditLog>;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['audit-logs'] });
    }
  });
}