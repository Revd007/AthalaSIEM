import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

interface CorrelationRule {
  id: string;
  name: string;
  conditions: any[];
  actions: any[];
  enabled: boolean;
}

export function useCorrelationRules() {
  return useQuery<CorrelationRule[]>({
    queryKey: ['correlation-rules'],
    queryFn: async () => {
      const response = await fetch('/correlation/rules');
      return response.json();
    }
  });
}

export function useCreateRule() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (rule: Partial<CorrelationRule>) => {
      const response = await fetch('/correlation/rules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rule)
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['correlation-rules'] });
    }
  });
}

export function useCorrelatedEvents(alertId: string) {
  return useQuery({
    queryKey: ['correlated-events', alertId],
    queryFn: async () => {
      const response = await fetch(`/correlation/events/${alertId}`);
      return response.json();
    }
  });
}