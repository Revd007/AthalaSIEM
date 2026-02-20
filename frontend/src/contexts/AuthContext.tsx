'use client'

import React, { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { authService, type LoginResponse } from '@/services/auth-service'

export interface AuthUser {
  id: string
  email: string
  username: string
  role: string
  full_name?: string
}

interface AuthContextValue {
  user: AuthUser | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (username: string, password: string) => Promise<LoginResponse>
  logout: () => Promise<void>
  refreshToken: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

const QUERY_KEYS_TO_INVALIDATE = [
  'events',
  'stats',
  'recent-alerts',
  'system-health',
  'threat-intelligence',
  'dashboard',
  'dashboard-summary',
  'agents',
  'logs',
  'alerts',
  'user',
  'ai',
  'threat-hunting',
]

function mapLoginUser(res: LoginResponse): AuthUser {
  const u = res.user
  return {
    id: String(u.id),
    email: u.email ?? '',
    username: u.username ?? '',
    role: u.role ?? 'User',
    full_name: u.full_name,
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const queryClient = useQueryClient()

  const refreshToken = useCallback(async () => {
    try {
      const u = await authService.getCurrentUser()
      setUser({
        id: String(u.id),
        email: u.email ?? '',
        username: u.username ?? '',
        role: u.role ?? 'User',
        full_name: u.full_name,
      })
    } catch {
      setUser(null)
    }
  }, [])

  useEffect(() => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null
    if (!token) {
      setUser(null)
      setIsLoading(false)
      return
    }
    authService
      .getCurrentUser()
      .then((u) => {
        setUser({
          id: String(u.id),
          email: u.email ?? '',
          username: u.username ?? '',
          role: u.role ?? 'User',
          full_name: u.full_name,
        })
      })
      .catch(() => setUser(null))
      .finally(() => setIsLoading(false))
  }, [])

  const login = useCallback(
    async (username: string, password: string) => {
      const res = await authService.login({ username, password })
      setUser(mapLoginUser(res))
      // Invalidate and refetch all AI & threat-hunting data so metrics don't stay at 0
      queryClient.invalidateQueries({
        predicate: (query) => {
          const key = query.queryKey[0]
          return (
            typeof key === 'string' &&
            (key === 'ai' || key === 'threat-hunting' || key === 'dashboard-summary' || key === 'dashboard')
          )
        },
      })
      return res
    },
    [queryClient]
  )

  const logout = useCallback(async () => {
    try {
      await authService.logout()
    } finally {
      setUser(null)
      queryClient.clear()
      await Promise.all(
        QUERY_KEYS_TO_INVALIDATE.map((key) =>
          queryClient.removeQueries({ queryKey: [key] })
        )
      )
      if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
        window.location.href = '/login'
      }
    }
  }, [queryClient])

  const value: AuthContextValue = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    logout,
    refreshToken,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuthContext(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuthContext must be used within AuthProvider')
  return ctx
}
