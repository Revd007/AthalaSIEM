'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Dashboard } from '@/components/Dashboard/Dashboard'

export default function DashboardPage() {
  const router = useRouter()

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    if (!token) {
      router.push('/login')
    }
  }, [router])

  return <Dashboard />
} 
