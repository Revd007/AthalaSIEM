// src/app/layout.tsx
import { Inter } from 'next/font/google'
import { Metadata } from 'next'
import './globals.css'
import Sidebar from '@/components/layout/sidebar'
import Navbar from '@/components/layout/navbar'
import Footer from '@/components/layout/footer'
import { Providers } from '@/components/providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SIEM Dashboard',
  description: 'Security Information and Event Management System',
}

interface RootLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="min-h-screen bg-gray-100">
            <div className="flex h-screen overflow-hidden">
              <Sidebar />
              <div className="flex flex-col flex-1 overflow-hidden">
                <Navbar />
                <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100">
                  {children}
                </main>
                <Footer />
              </div>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  )
}