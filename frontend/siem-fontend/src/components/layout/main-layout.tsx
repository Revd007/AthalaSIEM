import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { useAuth } from '../../hooks/use-auth'
import { 
  LayoutDashboard, 
  Bell, 
  Shield, 
  Settings, 
  LogOut 
} from 'lucide-react'

export function MainLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const { logout } = useAuth()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Alerts', href: '/alerts', icon: Bell },
    { name: 'Security', href: '/security', icon: Shield },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-white shadow-sm min-h-screen">
          <div className="p-4">
            <h1 className="text-2xl font-bold text-indigo-600">SIEM Dashboard</h1>
          </div>
          <nav className="mt-4">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center px-4 py-2 text-sm ${
                    router.pathname === item.href
                      ? 'bg-indigo-50 text-indigo-600'
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  {item.name}
                </Link>
              )
            })}
            <button
              onClick={() => logout()}
              className="flex items-center px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 w-full"
            >
              <LogOut className="w-5 h-5 mr-3" />
              Logout
            </button>
          </nav>
        </div>

        {/* Main content */}
        <div className="flex-1">
          <main className="p-6">{children}</main>
        </div>
      </div>
    </div>
  )
}