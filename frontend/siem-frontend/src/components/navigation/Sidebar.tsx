'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useAuthStore } from '../../hooks/use-auth'

interface NavItem {
  name: string
  href: string
  icon: string
  roles: string[]
}

const navigation: NavItem[] = [
  { 
    name: 'Dashboard', 
    href: '/dashboard', 
    icon: '📊',
    roles: ['admin', 'analyst', 'operator', 'viewer'] 
  },
  { 
    name: 'Alerts', 
    href: '/alerts', 
    icon: '🚨',
    roles: ['admin', 'analyst', 'operator'] 
  },
  { 
    name: 'Events', 
    href: '/events', 
    icon: '📋',
    roles: ['admin', 'analyst', 'operator'] 
  },
  { 
    name: 'Analytics', 
    href: '/analytics', 
    icon: '📈',
    roles: ['admin', 'analyst'] 
  },
  { 
    name: 'Playbooks', 
    href: '/playbooks', 
    icon: '📚',
    roles: ['admin', 'analyst'] 
  },
  { 
    name: 'Settings', 
    href: '/settings', 
    icon: '⚙️',
    roles: ['admin'] 
  },
  { 
    name: 'Users', 
    href: '/users', 
    icon: '👥',
    roles: ['admin'] 
  }
]

export default function Sidebar() {
  const pathname = usePathname()
  const { user } = useAuthStore()

  const userRole = user?.role?.toLowerCase() || 'viewer'

  const filteredNavigation = navigation.filter(item => 
    item.roles.includes(userRole)
  )

  return (
    <div className="w-64 bg-gray-800 min-h-screen p-4">
      <div className="mb-8">
        <h2 className="text-white text-xl font-bold">AthalaSIEM</h2>
      </div>
      
      <nav className="space-y-2">
        {filteredNavigation.map((item) => {
          const isActive = pathname === item.href
          
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`
                flex items-center px-4 py-2 text-sm rounded-lg
                ${isActive 
                  ? 'bg-gray-900 text-white' 
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }
              `}
            >
              <span className="mr-3">{item.icon}</span>
              {item.name}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}