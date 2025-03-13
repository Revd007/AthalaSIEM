import { Header } from '@/components/Header/Header'
import { Navigation } from '@/components/Header/Navigation'
import { FileCode } from 'lucide-react'

const sidebarItems = [
  {
    title: 'Playbooks',
    href: '/dashboard/playbooks',
    icon: FileCode,
  },
]

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <main className="flex-1 p-8">
        {children}
      </main>
    </div>
  )
} 