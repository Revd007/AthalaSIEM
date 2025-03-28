'use client'

import { useRouter } from 'next/navigation'
import { Bell, User, LogOut, Settings } from 'lucide-react'
import { toast } from 'sonner'
import { Navigation } from './Navigation'
import Image from 'next/image'
import { AlertNotifications } from './AlertNotifications'
import Link from 'next/link'

export function Header() {
  const router = useRouter()

  const handleLogout = () => {
    localStorage.removeItem('token')
    toast.success('Logged out successfully')
    router.push('/login')
  }

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
          <div className="flex items-center space-x-4">
            <AlertNotifications />
            
            <Link 
              href="/dashboard/settings"
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <Settings className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            </Link>
            
            <div className="relative">
              <button className="flex items-center space-x-3 p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700">
                <User className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Admin</span>
              </button>
            </div>

            <button 
              onClick={handleLogout}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              <LogOut className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}