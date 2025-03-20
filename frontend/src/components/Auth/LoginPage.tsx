'use client'

import React, { useState, useEffect } from 'react'
import { Shield, User, Lock, Eye, EyeOff } from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api } from '@/lib/api'
import { toast } from 'sonner'

// Add type definition for the API response
type AuthResponse = {
  token: string;
  refreshToken: string;
  user: {
    id: string;
    username: string;
    email: string;
    role?: string | string[];
  }
};

export function LoginPage() {
  const router = useRouter()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  // Combined effect for navigation
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      router.replace('/dashboard');
    }
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      // Basic validation
      if (!username || !password) {
        throw new Error('Please enter both username and password');
      }

      const response = await api.post<AuthResponse>('/api/auth/login', { 
        username, 
        password 
      })
      
      if (response.data?.token && response.data?.refreshToken) {
        // Store both tokens
        localStorage.setItem('token', response.data.token)
        localStorage.setItem('refreshToken', response.data.refreshToken)
        
        // Also store user info if available
        if (response.data.user) {
          console.debug('Login successful, storing user info:', response.data.user);
          localStorage.setItem('user', JSON.stringify(response.data.user));
          
          // Debug information for roles
          if (response.data.user.role) {
            console.debug('User roles:', response.data.user.role);
            // Show a more descriptive toast for roles
            const roleInfo = Array.isArray(response.data.user.role) 
              ? response.data.user.role.join(', ') 
              : response.data.user.role;
            toast.success(`Login successful as ${response.data.user.username} with role: ${roleInfo}`);
          } else {
            console.warn('No roles found in user info. User may not have permission for restricted areas.');
            toast.warning('Login successful but no role information found. You may have limited access.');
          }
        } else {
          toast.success('Login successful');
        }
        
        // Navigate directly after successful login
        router.replace('/dashboard');
      } else {
        throw new Error('Invalid response from server - missing token or refresh token')
      }
    } catch (error) {
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('user')
      console.error('Login error:', error);
      
      // Handle different types of errors
      let errorMessage = 'Login failed. Please check your credentials.';
      if (error instanceof Error) {
        if (error.message.includes('Failed to fetch')) {
          errorMessage = 'Unable to connect to the server. Please check your connection or try again.';
        } else {
          errorMessage = error.message;
        }
      }
      
      toast.error(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-400 to-blue-600">
      <div className="w-full max-w-md p-8 space-y-6 bg-white/10 backdrop-blur-lg rounded-2xl shadow-xl">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-white">Welcome back!</h1>
          <p className="mt-2 text-white/80">Please sign in to your account</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-4">
            <div className="relative">
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-white/50" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username"
                className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder:text-white/50 focus:outline-none focus:ring-2 focus:ring-white/50"
              />
            </div>

            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-white/50" />
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                className="w-full pl-10 pr-12 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder:text-white/50 focus:outline-none focus:ring-2 focus:ring-white/50"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/50"
              >
                {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-2 px-4 bg-white text-blue-600 rounded-lg font-medium hover:bg-white/90 focus:outline-none focus:ring-2 focus:ring-white/50 disabled:opacity-50"
          >
            {isLoading ? 'Signing In...' : 'Sign In'}
          </button>

          <div className="text-center">
            <Link href="/register" className="text-sm text-white hover:text-white/80">
              Don't have an account? Sign up
            </Link>
          </div>
        </form>
      </div>
    </div>
  )
}