import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { cn } from '../../lib/utils'

interface StatCardProps {
  title: string
  value: string
  icon: React.ReactNode
  description?: string
  variant?: 'default' | 'danger' | 'warning' | 'success'
  trend?: {
    direction: 'up' | 'down' | 'neutral'
    value: number
  }
}

export function StatCard({ 
  title, 
  value, 
  icon, 
  description, 
  variant = 'default',
  trend 
}: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "rounded-lg border bg-white p-6",
        variant === 'danger' && "border-red-200 bg-red-50",
        variant === 'warning' && "border-yellow-200 bg-yellow-50",
        variant === 'success' && "border-green-200 bg-green-50"
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-600">{title}</span>
        {icon && <div>{icon}</div>}
      </div>
      <div className="mt-2">
        <span className="text-2xl font-bold">{value}</span>
        {trend && (
          <span className={cn(
            "ml-2 text-sm",
            trend.direction === 'up' && "text-green-500",
            trend.direction === 'down' && "text-red-500",
            trend.direction === 'neutral' && "text-gray-500"
          )}>
            {trend.value}%
          </span>
        )}
      </div>
      {description && (
        <p className="text-sm text-gray-500">{description}</p>
      )}
    </motion.div>
  )
}