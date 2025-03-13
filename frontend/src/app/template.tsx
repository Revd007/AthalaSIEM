'use client'

import { useEffect, useState } from 'react'
import { Providers } from '@/components/ui/providers'
import { useAuthStore } from '@/hooks/use-auth'
import { QueryProvider } from '@/providers/query-provider'

export default function Template({
  children,
}: {
  children: React.ReactNode
}) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    useAuthStore.persist.rehydrate()
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="app-container">
      <QueryProvider>
        <Providers>
          {children}
        </Providers>
      </QueryProvider>
    </div>
  )
} 