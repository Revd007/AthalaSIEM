import axios, { AxiosInstance, AxiosRequestConfig } from 'axios'
import { useAuth } from '../hooks/use-auth'

class ApiService {
  private api: AxiosInstance
  private static instance: ApiService

  private constructor() {
    this.api = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  static getInstance(): ApiService {
    if (!ApiService.instance) {
      ApiService.instance = new ApiService()
    }
    return ApiService.instance
  }

  private setupInterceptors() {
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token')
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true
          try {
            const refreshToken = localStorage.getItem('refresh_token')
            const response = await this.refreshToken(refreshToken)
            localStorage.setItem('auth_token', response.data.token)
            return this.api(originalRequest)
          } catch (error) {
            useAuth.getState().logout()
            return Promise.reject(error)
          }
        }
        return Promise.reject(error)
      }
    )
  }

  private async refreshToken(refreshToken: string | null) {
    return await this.api.post('/auth/refresh-token', { refreshToken })
  }

  async get<T>(url: string, config?: AxiosRequestConfig) {
    const response = await this.api.get<T>(url, config)
    return response.data
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig) {
    const response = await this.api.post<T>(url, data, config)
    return response.data
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig) {
    const response = await this.api.put<T>(url, data, config)
    return response.data
  }

  async delete<T>(url: string, config?: AxiosRequestConfig) {
    const response = await this.api.delete<T>(url, config)
    return response.data
  }
}

export const apiService = ApiService.getInstance()