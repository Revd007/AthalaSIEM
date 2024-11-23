// src/hooks/use-auth.ts
import { create } from 'zustand'
import { axiosInstance } from '../lib/axios'

interface AuthState {
  user: any | null
  token: string | null
  isLoading: boolean
  login: (credentials: { username: string; password: string }) => Promise<void>
  logout: () => void
  checkAuth: () => Promise<void>
}

export const useAuth = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('token'),
  isLoading: true,
  
  login: async (credentials) => {
    try {
      const response = await axiosInstance.post('/auth/login', credentials)
      const { token, user } = response.data
      localStorage.setItem('token', token)
      set({ user, token })
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  },
  
  logout: () => {
    localStorage.removeItem('token')
    set({ user: null, token: null })
  },
  
  checkAuth: async () => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        set({ isLoading: false })
        return
      }
      
      const response = await axiosInstance.get('/auth/me')
      set({ user: response.data, isLoading: false })
    } catch (error) {
      localStorage.removeItem('token')
      set({ user: null, token: null, isLoading: false })
    }
  },
}))