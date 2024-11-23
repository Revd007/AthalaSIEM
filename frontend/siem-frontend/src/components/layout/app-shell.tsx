import React, { useState } from 'react'
import { useRouter } from 'next/router'
import {
  LayoutDashboard,
  Bell,
  Shield,
  Settings,
  Menu,
  X,
  ChevronDown,
  LogOut,
  User as UserIcon,
} from 'lucide-react'
import { useAuth } from '../../hooks/use-auth'
import { Button } from '../ui/button'
import { Avatar } from '../ui/avatar'
import { Dropdown, DropdownMenu, DropdownItem } from '../ui/dropdown'
import { User } from '../../types/user'

interface AppShellProps {
  children: React.ReactNode
}

export function AppShell({ children }: AppShellProps) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const router = useRouter()
  const { user, logout } = useAuth()

  const navigation = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: LayoutDashboard,
      items: [
        { name: 'Overview', href: '/dashboard/overview' },
        { name: 'Analytics', href: '/dashboard/analytics' },
        { name: 'Real-time', href: '/dashboard/real-time' },
      ],
    },
    {
      name: 'Alerts',
      href: '/alerts',
      icon: Bell,
      items: [
        { name: 'All Alerts', href: '/alerts' },
        { name: 'Critical', href: '/alerts/critical' },
        { name: 'Investigation', href: '/alerts/investigation' },
      ],
    },
    {
      name: 'Security',
      href: '/security',
      icon: Shield,
      items: [
        { name: 'Threats', href: '/security/threats' },
        { name: 'Compliance', href: '/security/compliance' },
        { name: 'Audit Logs', href: '/security/audit-logs' },
      ],
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Settings,
      items: [
        { name: 'General', href: '/settings/general' },
        { name: 'Notifications', href: '/settings/notifications' },
        { name: 'API', href: '/settings/api' },
      ],
    },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile Navigation Toggle */}
      <div className="lg:hidden fixed top-0 left-0 w-full bg-white z-50 px-4 py-3 flex items-center justify-between border-b">
        <Button
          variant="outline"
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        >
          {isSidebarOpen ? <X /> : <Menu />}
        </Button>
        <div className="flex items-center space-x-4">
          <Avatar
            src={user?.name}
            fallback={user?.name?.[0] || 'U'}
          />
        </div>
      </div>

      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 transform ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0 transition duration-200 ease-in-out z-30 w-64 bg-white border-r`}
      >
        <div className="h-full flex flex-col">
          {/* Logo */}
          <div className="p-4 border-b">
            <h1 className="text-xl font-bold text-primary-600">SIEM Dashboard</h1>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto py-4">
            {navigation.map((item) => (
              <div key={item.name} className="px-3 space-y-1">
                <button
                  className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded-md ${
                    router.pathname.startsWith(item.href)
                      ? 'bg-primary-50 text-primary-600'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center">
                    <item.icon className="w-5 h-5 mr-3" />
                    {item.name}
                  </div>
                  <ChevronDown className="w-4 h-4" />
                </button>
                {item.items && (
                  <div className="ml-8 space-y-1">
                    {item.items.map((subItem) => (
                      <button
                        key={subItem.name}
                        onClick={() => router.push(subItem.href)}
                        className={`w-full text-left px-3 py-2 text-sm rounded-md ${
                          router.pathname === subItem.href
                            ? 'bg-primary-50 text-primary-600'
                            : 'text-gray-600 hover:bg-gray-50'
                        }`}
                      >
                        {subItem.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </nav>

          {/* User Menu */}
          <div className="border-t p-4">
            <Dropdown>
              <div className="flex items-center space-x-3">
                <Avatar
                  src={user?.name}
                  fallback={user?.name?.[0] || 'U'}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {user?.name}
                  </p>
                  <p className="text-xs text-gray-500 truncate">
                    {user?.email}
                  </p>
                </div>
                <ChevronDown className="w-4 h-4 text-gray-500" />
              </div>
              <DropdownMenu>
                <DropdownItem icon={UserIcon} onClick={() => router.push('/profile')}>
                  Profile
                </DropdownItem>
                <DropdownItem icon={Settings} onClick={() => router.push('/settings')}>
                  Settings
                </DropdownItem>
                <DropdownItem icon={LogOut} onClick={logout}>
                  Logout
                </DropdownItem>
              </DropdownMenu>
            </Dropdown>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className={`lg:ml-64 min-h-screen pt-16 lg:pt-0`}>
        <main className="container mx-auto px-4 py-8">
          {children}
        </main>
      </div>
    </div>
  )
}