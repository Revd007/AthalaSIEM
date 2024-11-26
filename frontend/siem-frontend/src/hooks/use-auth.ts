import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { authService } from '../services/auth-service';
import type { LoginResponse } from '../services/auth-service';
import { useQuery, useMutation, QueryKey } from '@tanstack/react-query';
import { useEffect } from 'react';

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

  const loginMutation = useMutation({
    mutationFn: async (credentials: { username: string; password: string }) => {
      try {
        const response = await authService.login(credentials);
        useAuthStore.setState({ 
          user: response.user, 
          token: response.access_token,
          initialized: true
        });
        localStorage.setItem('token', response.access_token);
        return response;
      } catch (error: any) {
        console.error('Login error:', error);
        throw new Error(error.message || 'Login failed');
      }
    }
  });

  // Perbaikan query untuk getCurrentUser
  const {
    data: currentUser,
    isLoading: isLoadingUser,
  } = useQuery({
    queryKey: ['currentUser'],
    queryFn: () => authService.getCurrentUser(),
    enabled: !!token,
    retry: 1,
    select: (data: User) => {
      if (data) {
        setUser(data);
      }
      return data;
    },
    // Gunakan callbacks yang tersedia di versi terbaru
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });

  // Effect untuk handle error
  useEffect(() => {
    if (currentUser === undefined) {
      setToken(null);
      setUser(null);
    }
  }, [currentUser, setToken, setUser]);

  return {
    user: user || currentUser,
    isAuthenticated: !!token,
    isLoading: loginMutation.isPending || isLoadingUser,
    login: loginMutation.mutateAsync,
    logout: () => {
      setToken(null);
      setUser(null);
      localStorage.removeItem('token');
    },
    loginError: loginMutation.error
  };
}