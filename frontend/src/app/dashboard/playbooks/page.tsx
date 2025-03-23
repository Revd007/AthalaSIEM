'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { PlaybooksOverview } from '@/components/AutomatedPlaybooks/PlaybooksOverview'

export default function PlaybooksPage() {
  const router = useRouter()
  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    if (!token) {
      router.push('/login')
    }
  }, [router])
  return (
    <div className="container mx-auto p-6">
      <PlaybooksOverview />
    </div>
  )
} 