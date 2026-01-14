'use client'

import { ThemeProvider } from '@/components/theme-provider'
import { Toaster } from 'sonner'
import { useEffect, useState } from 'react'

export function Providers({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      {children}
      {mounted && <Toaster richColors closeButton position="top-right" />}
    </ThemeProvider>
  )
} 