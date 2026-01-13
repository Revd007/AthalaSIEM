'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Dashboard } from '@/components/Dashboard/Dashboard'

export default function DashboardPage() {
  const router = useRouter()
  const [isChecking, setIsChecking] = useState(true)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [mounted, setMounted] = useState(false)

  // Ensure we're on the client side before accessing localStorage
  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    // Only check authentication after component is mounted (client-side only)
    if (!mounted) return;
    
    // Prevent multiple checks - only run once
    if (!isChecking) return;

     // BREAKPOINT: Dashboard page checking authentication
    
    // Check if user is logged in
    const token = localStorage.getItem('token')
    const refreshToken = localStorage.getItem('refreshToken')
    
    console.log('[Dashboard] Checking authentication - Token exists:', !!token, 'Token length:', token?.length || 0, 'RefreshToken exists:', !!refreshToken, 'RefreshToken length:', refreshToken?.length || 0)
     // BREAKPOINT: After reading tokens from localStorage
    
    if (!token || !refreshToken) {
      console.log('[Dashboard] No tokens found, redirecting to login')
       // BREAKPOINT: About to redirect to login (tokens missing)
      // Only redirect once - set checking to false to prevent loop
      setIsChecking(false)
      router.replace('/login')
    } else {
      console.log('[Dashboard] Tokens found, allowing access')
       // BREAKPOINT: Tokens found, setting authenticated
      setIsAuthenticated(true)
      setIsChecking(false)
    }
  }, [mounted]) // Removed router from dependencies to prevent loop

  // Don't render dashboard until we've checked for token
  if (isChecking) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  // Only render dashboard if authenticated and mounted
  if (!mounted || !isAuthenticated) {
    if (!mounted) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Initializing...</p>
          </div>
        </div>
      )
    }
    return null
  }

  return <Dashboard />
} 
