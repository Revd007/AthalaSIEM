import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

interface APIKey {
  id: string;
  key: string;
  name: string;
  user_id: string;
  is_active: boolean;
  expires_at: string;
  created_at: string;
  last_used_at: string;
}

export function useAPIKeys() {
  return useQuery<APIKey[]>({
    queryKey: ['api-keys'],
    queryFn: async () => {
      const response = await fetch('/api-keys');
      return response.json();
    }
  });
}

export function useCreateAPIKey() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (name: string) => {
      const response = await fetch('/api-keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    }
  });
}

export function useRevokeAPIKey() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (id: string) => {
      await fetch(`/api-keys/${id}`, { method: 'DELETE' });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    }
  });
}