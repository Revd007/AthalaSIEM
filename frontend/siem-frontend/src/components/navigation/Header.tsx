'use client'

import { useState } from 'react'
import { useAuthStore } from '../../hooks/use-auth'
import { useRouter } from 'next/navigation'

export default function Header() {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const { user, logout } = useAuthStore()
  const router = useRouter()

  const handleLogout = () => {
    logout()
    router.push('/login')
  }

  return (
    <header className="bg-white shadow">
      <div className="mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between items-center">
          <div className="flex items-center">
            <h1 className="text-xl font-semibold">SIEM System</h1>
          </div>

          <div className="flex items-center">
            {/* Notifications */}
            <button className="p-2 text-gray-600 hover:text-gray-900">
              🔔
            </button>

            {/* Profile Dropdown */}
            <div className="relative ml-3">
              <button
                className="flex items-center gap-2 p-2 rounded-full hover:bg-gray-100"
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              >
                <span className="text-sm font-medium">{user?.username}</span>
                <span className="text-xs px-2 py-1 rounded-full bg-gray-100">
                  {user?.role}
                </span>
              </button>

              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10">
                  <button
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={() => router.push('/profile')}
                  >
                    Profile Settings
                  </button>
                  <button
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    onClick={handleLogout}
                  >
                    Sign out
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}