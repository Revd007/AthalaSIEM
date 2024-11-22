import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '../../lib/utils'

interface TimelineItem {
  id: string | number
  title: string
  description: string
  timestamp: string
  icon?: React.ReactNode
  status?: 'success' | 'warning' | 'error' | 'info'
}

interface TimelineProps {
  items: TimelineItem[]
  className?: string
}

export function Timeline({ items, className }: TimelineProps) {
  const statusColors = {
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
    info: 'bg-blue-500',
  }

  return (
    <div className={cn('relative space-y-8', className)}>
      {items.map((item, index) => (
        <motion.div
          key={item.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="relative pl-8"
        >
          <div
            className={cn(
              'absolute left-0 flex h-6 w-6 items-center justify-center rounded-full',
              item.status ? statusColors[item.status] : 'bg-gray-200'
            )}
          >
            {item.icon}
          </div>
          <div className="flex flex-col space-y-1">
            <div className="flex items-center space-x-2">
              <h4 className="font-medium text-gray-900">{item.title}</h4>
              <span className="text-sm text-gray-500">{item.timestamp}</span>
            </div>
            <p className="text-sm text-gray-600">{item.description}</p>
          </div>
          {index !== items.length - 1 && (
            <div className="absolute left-3 top-6 h-full w-px bg-gray-200" />
          )}
        </motion.div>
      ))}
    </div>
  )
}