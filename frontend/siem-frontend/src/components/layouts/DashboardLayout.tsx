'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../hooks/use-auth'
import Sidebar from '../../components/navigation/Sidebar'
import Header from '../../components/navigation/Header'

export default function DashboardLayout({
  children
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const { user, initialized } = useAuthStore()

  useEffect(() => {
    if (initialized && !user) {
      router.push('/login')
    }
  }, [user, initialized, router])

  if (!user) {
    return null
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6">
          {children}
        </main>
      </div>
    </div>
  )
}