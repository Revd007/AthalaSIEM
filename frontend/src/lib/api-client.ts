import { createTRPCProxyClient, httpBatchLink } from '@trpc/client';
import type { AppRouter } from './trpc';

export const api = createTRPCProxyClient<AppRouter>({
  links: [
    httpBatchLink({
      url: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9595/api/trpc',
      headers: () => {
        const token = localStorage.getItem('auth_token');
        return {
          Authorization: token ? `Bearer ${token}` : '',
        };
      },
    }),
  ],
});

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9595';

export async function fetchApi(endpoint: string, options?: RequestInit) {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }

  return response.json();
} 