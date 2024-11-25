// src/hooks/use-auth.ts
import { create } from 'zustand'
import { axiosInstance } from '../lib/axios'
import { persist, createJSONStorage } from 'zustand/middleware'

// Simplified state interface
interface AuthState {
  user: any | null;
  token: string | null;
  initialized: boolean;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
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
      login: async (username: string, password: string) => {
        try {
          set({ loading: true });
          const response = await axiosInstance.post('/auth/login', {
            username,
            password,
          });
          
          const { access_token, user } = response.data;
          
          // Store token in localStorage and set axios default header
          localStorage.setItem('token', access_token);
          axiosInstance.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          
          set({ 
            token: access_token, 
            user, 
            loading: false, 
            initialized: true 
          });

          return response.data;
        } catch (error: any) {
          set({ loading: false });
          throw new Error(error.response?.data?.detail || 'Login failed');
        }
      },
      logout: () => {
        localStorage.removeItem('token');
        delete axiosInstance.defaults.headers.common['Authorization'];
        set({ user: null, token: null });
      }
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      skipHydration: true
    }
  )
)

