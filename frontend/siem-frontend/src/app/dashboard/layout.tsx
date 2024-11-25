'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../hooks/use-auth'
import dynamic from 'next/dynamic'

const Sidebar = dynamic(() => import('../../components/layout/sidebar'), {
  loading: () => <div>Loading...</div>,
  ssr: false
})

const Navbar = dynamic(() => import('../../components/layout/navbar'), {
  loading: () => <div>Loading...</div>,
  ssr: false
})

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const { token, initialized, loading } = useAuthStore()

  useEffect(() => {
    if (initialized && !token) {
      router.replace('/login')
    }
  }, [token, initialized, router])

  // Show loading state
  if (loading || !initialized) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="w-8 h-8 border-2 border-primary rounded-full animate-spin border-t-transparent"></div>
      </div>
    )
  }

  // If no token, don't render anything
  if (!token) return null

  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  )
}