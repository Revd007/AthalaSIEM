'use client'

import { Inter } from 'next/font/google'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { ThemeProvider } from '@/components/theme-provider'
import { Toaster } from 'sonner'
import { useEffect } from 'react'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60,
      refetchOnWindowFocus: false,
    },
  },
})

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
        <QueryClientProvider client={queryClient}>
          <ThemeProvider 
            attribute="class" 
            defaultTheme="system" 
            enableSystem 
            disableTransitionOnChange
          >
            {children}
            <Toaster richColors closeButton position="top-right" />
          </ThemeProvider>
          <ReactQueryDevtools initialIsOpen={false} />
        </QueryClientProvider>
      </body>
    </html>
  )
} 