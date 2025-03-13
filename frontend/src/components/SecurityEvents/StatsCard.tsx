import { LucideIcon } from 'lucide-react'

interface StatsCardProps {
  title: string
  value: string
  change: string
  icon: LucideIcon
  trend: 'up' | 'down'
  color?: 'blue' | 'red' | 'green' | 'yellow'
}

export function StatsCard({ 
  title, 
  value, 
  change, 
  icon: Icon, 
  trend, 
  color = 'blue' 
}: StatsCardProps) {
  const colors = {
    blue: {
      bg: 'bg-blue-50 dark:bg-blue-900/20',
      text: 'text-blue-600 dark:text-blue-400',
      icon: 'text-blue-500',
    },
    red: {
      bg: 'bg-red-50 dark:bg-red-900/20',
      text: 'text-red-600 dark:text-red-400',
      icon: 'text-red-500',
    },
    green: {
      bg: 'bg-green-50 dark:bg-green-900/20',
      text: 'text-green-600 dark:text-green-400',
      icon: 'text-green-500',
    },
    yellow: {
      bg: 'bg-yellow-50 dark:bg-yellow-900/20',
      text: 'text-yellow-600 dark:text-yellow-400',
      icon: 'text-yellow-500',
    },
  }

  return (
    <div className={`${colors[color].bg} rounded-lg p-6`}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
          {title}
        </span>
        <Icon className={`h-5 w-5 ${colors[color].icon}`} />
      </div>
      <div className="mt-4">
        <span className="text-2xl font-bold text-gray-900 dark:text-white">
          {value}
        </span>
        <span className={`
          ml-2 text-sm font-medium
          ${trend === 'up' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}
        `}>
          {change}
        </span>
      </div>
    </div>
  )
} 