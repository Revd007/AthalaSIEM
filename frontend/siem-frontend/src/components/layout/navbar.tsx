import { useState } from 'react'
import { BellIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline'
import { UserCircleIcon } from '@heroicons/react/24/solid'
import React from 'react'

const Navbar = () => {
  const [notifications] = useState([
    { id: 1, message: 'Critical alert detected' },
    { id: 2, message: 'New security update available' },
  ])

  return (
    <header className="bg-white shadow">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between">
          <div className="flex flex-1">
            <div className="flex w-full md:ml-0">
              <div className="relative w-full">
                <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                  <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
                </div>
                <input
                  type="search"
                  className="block w-full rounded-md border-0 py-1.5 pl-10 pr-3 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                  placeholder="Search logs, alerts..."
                />
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="relative">
              <button
                type="button"
                className="relative rounded-full bg-white p-1 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 text-xs text-white flex items-center justify-center">
                  {notifications.length}
                </span>
                <BellIcon className="h-6 w-6" aria-hidden="true" />
              </button>
            </div>
            
            <div className="relative">
              <button
                type="button"
                className="flex items-center gap-2 rounded-full bg-white p-1 text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                <UserCircleIcon className="h-8 w-8" aria-hidden="true" />
                <span className="text-sm font-medium text-gray-700">Admin</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Navbar