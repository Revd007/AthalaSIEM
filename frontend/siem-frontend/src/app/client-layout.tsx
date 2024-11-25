'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Providers } from '../components/ui/providers'

const queryClient = new QueryClient()

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <QueryClientProvider client={queryClient}>
      <Providers>{children}</Providers>
    </QueryClientProvider>
  )
}