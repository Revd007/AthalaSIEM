'use client'

import { Inter } from 'next/font/google'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { queryClient, persistOptions } from '@/lib/query-client'
import { ThemeProvider } from '@/components/theme-provider'
import { Toaster } from 'sonner'
import { useEffect } from 'react'
import { AuthProvider } from '@/contexts/AuthContext'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Run token debugger on initial load - only in development
  useEffect(() => {
    if (process.env.NODE_ENV !== 'production' && typeof window !== 'undefined') {
      // Import and initialize token debugger
      import('@/debug/token-debugger').then(mod => {
        console.log('Token Debugger initialized - run debugToken() in console to analyze JWT tokens');
        // Make the debugger available globally
        (window as any).debugToken = mod.debugToken;
      }).catch(err => {
        console.error('Failed to load token debugger:', err);
      });
    }
  }, []);

  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <PersistQueryClientProvider client={queryClient} persistOptions={persistOptions}>
          <AuthProvider>
            <ThemeProvider 
              attribute="class" 
              defaultTheme="system" 
              enableSystem 
              disableTransitionOnChange
            >
              {children}
              <Toaster richColors closeButton position="top-right" />
            </ThemeProvider>
          </AuthProvider>
          <ReactQueryDevtools initialIsOpen={false} />
        </PersistQueryClientProvider>
      </body>
    </html>
  )
} 