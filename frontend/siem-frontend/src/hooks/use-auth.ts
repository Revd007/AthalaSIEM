// src/hooks/use-auth.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import api from '../lib/api'

interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'analyst' | 'user'
}

interface AuthState {
  user: User | null
  token: string | null
  isLoading: boolean
  error: string | null
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  setUser: (user: User) => void
}

export const useAuth = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isLoading: false,
      error: null,
      login: async (email: string, password: string) => {
        try {
          set({ isLoading: true, error: null })
          const response = await api.post('/auth/login', { email, password })
          const { user, token } = response.data
          
          set({ user, token, isLoading: false })
          api.defaults.headers.common.Authorization = `Bearer ${token}`
        } catch (error) {
          set({ 
            error: 'Invalid credentials', 
            isLoading: false 
          })
        }
      },
      logout: () => {
        set({ user: null, token: null })
        delete api.defaults.headers.common.Authorization
      },
      setUser: (user: User) => set({ user })
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token }),
    }
  )
)