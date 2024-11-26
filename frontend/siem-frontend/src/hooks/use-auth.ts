import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { authService } from '../services/auth-service';
import { useQuery, useMutation } from '@tanstack/react-query';

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
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  setInitialized: (initialized: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      initialized: false,
      loading: false,
      setInitialized: (initialized) => set({ initialized }),
      login: async (email: string, password: string) => {
        try {
          set({ loading: true });
          const response = await authService.login({ email, password });
          
          const { access_token, user } = response;
          
          if (!access_token || !user) {
            throw new Error('Invalid response from server');
          }
          
          localStorage.setItem('token', access_token);
          
          set({ 
            token: access_token, 
            user, 
            loading: false, 
            initialized: true 
          });
          
        } catch (error: any) {
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
  const { user, token, login, logout, loading } = useAuthStore();

  const loginMutation = useMutation({
    mutationFn: (credentials: { email: string; password: string }) => 
      authService.login(credentials),
    onSuccess: (data) => {
      useAuthStore.setState({ 
        user: data.user, 
        token: data.access_token,
        initialized: true
      });
      localStorage.setItem('token', data.access_token);
    }
  });

  const { data: currentUser, isLoading: isLoadingUser } = useQuery({
    queryKey: ['currentUser'],
    queryFn: authService.getCurrentUser,
    enabled: !!token,
    retry: false
  });

  return {
    user: user || currentUser,
    isAuthenticated: !!token,
    isLoading: loading || isLoadingUser,
    login: loginMutation.mutate,
    logout,
    loginError: loginMutation.error
  };
}