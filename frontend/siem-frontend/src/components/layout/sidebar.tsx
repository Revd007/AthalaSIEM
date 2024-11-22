import Link from 'next/link'
import { 
  HomeIcon, 
  DocumentTextIcon,
  BellIcon,
  ChartBarIcon,
  Cog6ToothIcon 
} from '@heroicons/react/24/outline'
import React from 'react'

const navigation = [
  { name: 'Overview', href: '/overview', icon: HomeIcon },
  { name: 'Logs', href: '/logs', icon: DocumentTextIcon },
  { name: 'Alerts', href: '/alerts', icon: BellIcon },
  { name: 'Reports', href: '/reports', icon: ChartBarIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
]

const Sidebar = () => {
  return (
    <div className="hidden md:flex md:w-64 md:flex-col">
      <div className="flex flex-col flex-grow pt-5 bg-white overflow-y-auto">
        <div className="flex items-center flex-shrink-0 px-4">
          <img
            className="h-8 w-auto"
            src="/images/logo.svg"
            alt="SIEM Logo"
          />
        </div>
        <div className="mt-5 flex-1 flex flex-col">
          <nav className="flex-1 px-2 space-y-1">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="group flex items-center px-2 py-2 text-sm font-medium rounded-md text-gray-600 hover:bg-gray-50 hover:text-gray-900"
              >
                <item.icon
                  className="mr-3 flex-shrink-0 h-6 w-6"
                  aria-hidden="true"
                />
                {item.name}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </div>
  )
}

export default Sidebar