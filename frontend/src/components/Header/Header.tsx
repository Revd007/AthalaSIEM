'use client'

import { useRouter } from 'next/navigation'
import { Bell, User, LogOut, Settings, Key, Shield, Users, ChevronDown } from 'lucide-react'
import { toast } from 'sonner'
import { Navigation } from './Navigation'
import Image from 'next/image'
import { AlertNotifications } from './AlertNotifications'
import Link from 'next/link'
import { useState, useEffect, useRef } from 'react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

interface UserInfo {
  username: string
  email: string
  role: string
  roles?: string[]
}

export function Header() {
  const router = useRouter()
  const [user, setUser] = useState<UserInfo | null>(null)

  useEffect(() => {
    // Get user info from localStorage or token
    const storedUser = localStorage.getItem('user')
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser))
      } catch {
        setUser({ username: 'Admin', email: '', role: 'Admin' })
      }
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    toast.success('Logged out successfully')
    router.push('/login')
  }

  const isAdmin = user?.role === 'Admin' || user?.roles?.includes('Admin')

  return (
    <header className="bg-white shadow-sm dark:bg-gray-800 sticky top-0 z-40">
      <div className="max-w-full mx-auto px-4 sm:px-6">
        <div className="flex h-16 items-center justify-between">
          {/* Logo & Navigation */}
          <div className="flex items-center flex-1">
            <div className="flex-shrink-0">
              <Image
                src="/logo.png"
                alt="Athala SIEM"
                width={40}
                height={40}
                className="h-8 w-auto"
              />
            </div>
            <div className="ml-6 flex-1 overflow-x-auto">
              <Navigation />
            </div>
          </div>

          {/* Right side buttons */}
          <div className="flex items-center space-x-2 sm:space-x-4">
            <AlertNotifications />
            
            <Link 
              href="/dashboard/settings"
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <Settings className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            </Link>
            
            {/* Profile Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  <div className="h-8 w-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <span className="text-sm font-semibold text-white">
                      {(user?.username?.[0] || 'A').toUpperCase()}
                    </span>
                  </div>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300 hidden sm:block">
                    {user?.username || 'Admin'}
                  </span>
                  <ChevronDown className="h-4 w-4 text-gray-500 hidden sm:block" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>
                  <div className="flex flex-col">
                    <span className="font-medium">{user?.username || 'Admin'}</span>
                    <span className="text-xs text-gray-500">{user?.email || ''}</span>
                    <span className="text-xs text-blue-600 mt-1">{user?.role || 'Admin'}</span>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link href="/dashboard/profile" className="cursor-pointer">
                    <User className="h-4 w-4 mr-2" />
                    My Profile
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/dashboard/profile?tab=security" className="cursor-pointer">
                    <Key className="h-4 w-4 mr-2" />
                    Change Password
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/dashboard/profile?tab=security" className="cursor-pointer">
                    <Shield className="h-4 w-4 mr-2" />
                    Two-Factor Auth
                  </Link>
                </DropdownMenuItem>
                {isAdmin && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem asChild>
                      <Link href="/dashboard/admin/users" className="cursor-pointer">
                        <Users className="h-4 w-4 mr-2" />
                        User Management
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem asChild>
                      <Link href="/dashboard/admin/security" className="cursor-pointer">
                        <Shield className="h-4 w-4 mr-2" />
                        Security Settings
                      </Link>
                    </DropdownMenuItem>
                  </>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleLogout} className="text-red-600 cursor-pointer">
                  <LogOut className="h-4 w-4 mr-2" />
                  Sign Out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </header>
  )
}