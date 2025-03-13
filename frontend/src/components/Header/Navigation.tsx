'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  Home, 
  Shield,
  Target,
  Brain, 
  Activity,
  PlayCircle,
  AlertTriangle,
  Users,
  FileCheck,
  Network,
  Settings,
  Menu,
  X
} from 'lucide-react'
import React from 'react'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home },
  { name: 'Security Events', href: '/dashboard/events', icon: Shield },
  { name: 'Threat Hunting', href: '/dashboard/threat-hunting', icon: Target },
  { name: 'AI Analysis', href: '/dashboard/ai-analysis', icon: Brain },
  { name: 'Predictive Analytics', href: '/dashboard/predictive', icon: Activity },
  { name: 'Automated Playbooks', href: '/dashboard/playbooks', icon: PlayCircle },
  { name: 'Active Incidents', href: '/dashboard/incidents', icon: AlertTriangle },
  { name: 'Team Collaboration', href: '/dashboard/collaboration', icon: Users },
  { name: 'Compliance', href: '/dashboard/compliance', icon: FileCheck },
  { name: 'System Health', href: '/dashboard/health', icon: Network },
  { name: 'Settings', href: '/dashboard/settings', icon: Settings },
]

export function Navigation() {
  const pathname = usePathname()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="hidden lg:flex items-center space-x-1">
        {navigation.map((item) => {
          const isActive = pathname === item.href
          const Icon = item.icon
          
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`
                inline-flex items-center px-3 py-2 text-sm font-medium rounded-lg
                transition-colors duration-200 whitespace-nowrap
                ${isActive 
                  ? 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400' 
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-800'
                }
              `}
            >
              <Icon className={`
                mr-2 h-5 w-5 
                ${isActive ? 'text-blue-500' : 'text-gray-400'}
              `} />
              {item.name}
            </Link>
          )
        })}
      </nav>

      {/* Mobile Menu Button */}
      <div className="lg:hidden">
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-800"
        >
          {isMobileMenuOpen ? (
            <X className="h-6 w-6" />
          ) : (
            <Menu className="h-6 w-6" />
          )}
        </button>
      </div>

      {/* Mobile Navigation */}
      {isMobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 z-50 bg-white dark:bg-gray-800">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Menu</h2>
              <button
                onClick={() => setIsMobileMenuOpen(false)}
                className="p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-800"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            <nav className="space-y-2">
              {navigation.map((item) => {
                const isActive = pathname === item.href
                const Icon = item.icon
                
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`
                      flex items-center px-4 py-3 text-sm font-medium rounded-lg w-full
                      transition-colors duration-200
                      ${isActive 
                        ? 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400' 
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-800'
                      }
                    `}
                  >
                    <Icon className={`
                      mr-3 h-5 w-5 
                      ${isActive ? 'text-blue-500' : 'text-gray-400'}
                    `} />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      )}
    </>
  )
}