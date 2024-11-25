// src/app/layout.tsx
'use client'

import { useEffect, useState } from 'react'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '../components/ui/providers'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useAuthStore } from '../hooks/use-auth'
import { HashLoader } from 'react-spinners'

const inter = Inter({ subsets: ['latin'] })
const queryClient = new QueryClient()

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const [mounted, setMounted] = useState(false)
  const setInitialized = useAuthStore(state => state.setInitialized)

  useEffect(() => {
    setMounted(true)
    useAuthStore.persist.rehydrate()
    setInitialized(true)
  }, [setInitialized])

  if (!mounted) {
    return (
      <html lang="en">
        <body className={inter.className}>
          <div className="min-h-screen flex items-center justify-center">
            <HashLoader color="#000B58" size={50} />
          </div>
        </body>
      </html>
    )
  }

  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <QueryClientProvider client={queryClient}>
          <Providers>
            {children}
          </Providers>
        </QueryClientProvider>
      </body>
    </html>
  )
}