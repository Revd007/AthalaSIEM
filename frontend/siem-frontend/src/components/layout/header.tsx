import React from 'react'
import Image from 'next/image'
import { motion } from 'framer-motion'
import { Bell, Settings, User } from 'lucide-react'
import { NotificationCenter } from '../ui/notification-center'

export function Header() {
  return (
    <header className="fixed top-0 z-50 w-full border-b bg-white">
      <div className="flex h-16 items-center justify-between px-4 md:px-6">
        {/* Logo dan Brand */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center space-x-3"
        >
          <Image
            src="/logo-athala.png"
            alt="AthalaSIEM Logo"
            width={40}
            height={40}
          />
          <div>
            <h1 className="text-xl font-bold text-primary-600">AthalaSIEM</h1>
            <p className="text-xs text-gray-500">Security Information and Event Management</p>
          </div>
        </motion.div>

        {/* Navigation Actions */}
        <div className="flex items-center space-x-4">
          <NotificationCenter />
          <button className="rounded-full p-2 hover:bg-gray-100">
            <Settings className="h-5 w-5 text-gray-600" />
          </button>
          <div className="flex items-center space-x-2">
            <div className="text-right hidden md:block">
              <p className="text-sm font-medium">Admin User</p>
              <p className="text-xs text-gray-500">admin@athala.com</p>
            </div>
            <button className="rounded-full bg-gray-100 p-2">
              <User className="h-5 w-5 text-gray-600" />
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}