import axios from 'axios'

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Add response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export const endpoints = {
  auth: {
    login: '/auth/login',
    logout: '/auth/logout',
    register: '/auth/register',
  },
  logs: {
    getAll: '/logs',
    getById: (id: string) => `/logs/${id}`,
    search: '/logs/search',
  },
  alerts: {
    getAll: '/alerts',
    getById: (id: string) => `/alerts/${id}`,
    create: '/alerts',
    update: (id: string) => `/alerts/${id}`,
  },
  reports: {
    generate: '/reports/generate',
    getAll: '/reports',
  },
}

export default api