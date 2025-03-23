import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

interface User {
  id: string
  email: string
  username: string
}

interface AuthResponse {
  token: string;
  refreshToken: string;
  user: User;
}

interface AuthState {
  user: User | null
  isLoading: boolean
  login: (credentials: { username: string; password: string }) => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isLoading: true,
      login: async (credentials) => {
        try {
          const response = await api.post('/api/auth/login', credentials)
          const authData = response.data as AuthResponse
          set({ user: authData.user, isLoading: false })
          localStorage.setItem('token', authData.token)
          localStorage.setItem('refreshToken', authData.refreshToken)
        } catch (error) {
          console.error('Login failed:', error)
          throw error
        }
      },
      logout: () => {
        localStorage.removeItem('token')
        localStorage.removeItem('refreshToken')
        set({ user: null })
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ user: state.user })
    }
  )
)

export function useAuth() {
  const { data: user, isLoading } = useQuery<User>({
    queryKey: ['user'],
    queryFn: async () => {
      try {
        const response = await fetch('/api/auth/me')
        if (!response.ok) throw new Error('Failed to fetch user')
        return response.json()
      } catch (error) {
        if (error instanceof TypeError && error.message === 'Failed to fetch') {
          return null
        }
        throw error
      }
    },
    retry: false,
    refetchOnWindowFocus: false
  })

  const loginMutation = useMutation({
    mutationFn: async (credentials: { username: string; password: string }) => {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
      })
      if (!response.ok) throw new Error('Login failed')
      return response.json()
    }
  })

  return {
    user,
    isLoading,
    login: loginMutation.mutate,
    isLoginLoading: loginMutation.isPending
  }
} 