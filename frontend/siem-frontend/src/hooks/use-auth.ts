import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { authService } from '../services/auth-service';
import type { LoginResponse } from '../services/auth-service';
import { useQuery, useMutation, QueryKey } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

interface User {
  id: string;
  email: string;
  username: string;
  role: string;
  full_name?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  initialized: boolean;
  loading: boolean;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  setInitialized: (initialized: boolean) => void;
  login: (credentials: { username: string; password: string }) => Promise<LoginResponse>;
  logout: () => void;
}

export const useAuthStore = create(
  persist<AuthState>(
    (set) => ({
      user: null,
      token: null,
      initialized: false,
      loading: false,
      setUser: (user) => set({ user }),
      setToken: (token) => set({ token }),
      setInitialized: (initialized) => set({ initialized }),
      login: async (credentials) => {
        try {
          set({ loading: true });
          const response = await authService.login(credentials);
          set({ 
            user: response.user, 
            token: response.access_token,
            initialized: true 
          });
          return response;
        } catch (error) {
          set({ loading: false });
          throw error;
        }
      },
      logout: () => {
        localStorage.removeItem('token');
        set({ user: null, token: null });
      }
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      skipHydration: true
    }
  )
);

// Hook untuk menggunakan auth dengan React Query
export function useAuth() {
  const { user, token, setUser, setToken } = useAuthStore();
  const [isLoading, setIsLoading] = useState(true);
  
  const loginMutation = useMutation({
    mutationFn: async (credentials: { username: string; password: string }) => {
      const response = await authService.login(credentials);
      // Set cookie and localStorage
      document.cookie = `token=${response.access_token}; path=/;`;
      localStorage.setItem('token', response.access_token);
      
      // Update store
      useAuthStore.setState({ 
        user: response.user, 
        token: response.access_token,
        initialized: true 
      });
      
      return response;
    }
  });

  // Check auth status on mount
  useEffect(() => {
    const checkAuth = async () => {
      setIsLoading(true);
      try {
        // Check both cookie and localStorage
        const cookieToken = document.cookie
          .split('; ')
          .find(row => row.startsWith('token='))
          ?.split('=')[1];
        
        const localToken = localStorage.getItem('token');
        
        // If no token anywhere, clear auth state
        if (!cookieToken && !localToken) {
          setToken(null);
          setUser(null);
          setIsLoading(false);
          return;
        }

        // Use token to verify with backend
        const response = await authService.getCurrentUser();
        if (response) {
          useAuthStore.setState({ 
            user: response,
            token: cookieToken || localToken,
            initialized: true 
          });
        } else {
          // Clear invalid auth state
          setToken(null);
          setUser(null);
          localStorage.removeItem('token');
          document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT;';
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        // Clear auth state on error
        setToken(null);
        setUser(null);
        localStorage.removeItem('token');
        document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT;';
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [setToken, setUser]);

  return {
    user,
    token,
    isAuthenticated: !!token && !!user,
    isLoading: isLoading || loginMutation.isPending,
    login: loginMutation.mutateAsync,
    logout: async () => {
      try {
        await authService.logout();
      } finally {
        document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT;';
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
        useAuthStore.setState({ initialized: true });
        window.location.href = '/login';
      }
    },
    loginError: loginMutation.error
  };
}