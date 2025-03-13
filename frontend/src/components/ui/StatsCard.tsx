'use client'

import { LucideIcon } from 'lucide-react'
import { Card } from '@/components/ui/card'

interface StatsCardProps {
  title: string
  value: string
  change?: string
  icon: LucideIcon
  trend?: 'up' | 'down' | 'neutral'
  color?: 'green' | 'red' | 'yellow' | 'blue'
}

const colorConfig = {
  green: {
    icon: 'text-green-500',
    bg: 'bg-green-50 dark:bg-green-900/20',
    text: 'text-green-700 dark:text-green-300',
  },
  red: {
    icon: 'text-red-500',
    bg: 'bg-red-50 dark:bg-red-900/20',
    text: 'text-red-700 dark:text-red-300',
  },
  yellow: {
    icon: 'text-yellow-500',
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    text: 'text-yellow-700 dark:text-yellow-300',
  },
  blue: {
    icon: 'text-blue-500',
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    text: 'text-blue-700 dark:text-blue-300',
  },
}

export function StatsCard({ 
  title, 
  value, 
  change, 
  icon: Icon,
  trend = 'neutral',
  color = 'blue'
}: StatsCardProps) {
  return (
    <Card>
      <div className="p-6">
        <div className="flex items-center justify-between">
          <div className={`p-2 rounded-lg ${colorConfig[color].bg}`}>
            <Icon className={`h-5 w-5 ${colorConfig[color].icon}`} />
          </div>
          {change && (
            <span className={`
              text-sm font-medium
              ${trend === 'up' ? 'text-green-600 dark:text-green-400' :
                trend === 'down' ? 'text-red-600 dark:text-red-400' :
                'text-gray-600 dark:text-gray-400'}
            `}>
              {change}
            </span>
          )}
        </div>
        <div className="mt-4">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
            {title}
          </h3>
          <p className="mt-2 text-3xl font-semibold text-gray-900 dark:text-white">
            {value}
          </p>
        </div>
      </div>
    </Card>
  )
} 