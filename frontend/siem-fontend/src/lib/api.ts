// src/lib/api.ts
import axios from 'axios'
import { useAuth } from '@/hooks/use-auth'

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use((config) => {
  const token = useAuth.getState().token
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      useAuth.getState().logout()
    }
    return Promise.reject(error)
  }
)

export const endpoints = {
  auth: {
    login: '/auth/login',
    register: '/auth/register',
    logout: '/auth/logout',
    refresh: '/auth/refresh-token',
  },
  alerts: {
    list: '/alerts',
    create: '/alerts',
    update: (id: string) => `/alerts/${id}`,
    delete: (id: string) => `/alerts/${id}`,
  },
  logs: {
    list: '/logs',
    search: '/logs/search',
  },
  reports: {
    generate: '/reports/generate',
    list: '/reports',
    download: (id: string) => `/reports/${id}/download`,
  },
}

export default api