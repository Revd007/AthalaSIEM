'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../hooks/use-auth'
import Sidebar from '../../components/navigation/Sidebar'
import Header from '../../components/navigation/Header'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const { user, initialized } = useAuthStore()

  useEffect(() => {
    if (initialized && !user) {
      router.push('/login')
    }
  }, [initialized, user, router])

  if (!user) return null

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-200">
          <div className="container mx-auto px-6 py-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}