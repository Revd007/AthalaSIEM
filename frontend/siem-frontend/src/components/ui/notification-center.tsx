import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bell, X, Check, AlertCircle } from 'lucide-react'
import { Button } from './button'
import { cn } from '../../lib/utils'

export interface Notification {
  id: string | number
  title: string
  message: string
  type: string
  timestamp: Date
  read: boolean
}

interface NotificationCenterProps {
  notifications: Notification[]
  onMarkAsRead: (id: string) => void
  onClearAll: () => void
}

export function NotificationCenter({
  notifications,
  onMarkAsRead,
  onClearAll,
}: NotificationCenterProps) {
  const [isOpen, setIsOpen] = useState(false)

  const getIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return <Check className="h-5 w-5 text-green-500" />
      case 'warning':
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />
      default:
        return <Bell className="h-5 w-5 text-blue-500" />
    }
  }

  const unreadCount = notifications.filter((n) => !n.read).length

  return (
    <div className="relative">
      <Button
        variant="outline"
        onClick={() => setIsOpen(!isOpen)}
        className="relative h-9 w-9 p-0"
      >
        <Bell className="h-5 w-5" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-xs text-white">
            {unreadCount}
          </span>
        )}
      </Button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute right-0 mt-2 w-80 rounded-lg border bg-white shadow-lg"
          >
            <div className="flex items-center justify-between border-b p-4">
              <h3 className="font-semibold">Notifications</h3>
              <Button
                variant="outline"
                onClick={onClearAll}
              >
                Clear all
              </Button>
            </div>

            <div className="max-h-[400px] overflow-y-auto">
              {notifications.length === 0 ? (
                <p className="p-4 text-center text-sm text-gray-500">
                  No notifications
                </p>
              ) : (
                notifications.map((notification) => (
                  <motion.div
                    key={notification.id}
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className={cn(
                      'flex items-start p-4 hover:bg-gray-50',
                      !notification.read && 'bg-blue-50'
                    )}
                  >
                    <div className="mr-3 mt-1">{getIcon(notification.type)}</div>
                    <div className="flex-1">
                      <h4 className="text-sm font-medium">
                        {notification.title}
                      </h4>
                      <p className="mt-1 text-sm text-gray-500">
                        {notification.message}
                      </p>
                      <p className="mt-1 text-xs text-gray-400">
                        {notification.timestamp.toLocaleString()}
                      </p>
                    </div>
                    {!notification.read && (
                      <Button
                        variant="outline"
                        onClick={() => onMarkAsRead(notification.id.toString())}
                      >
                        Mark as read
                      </Button>
                    )}
                  </motion.div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}