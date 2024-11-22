import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '../../lib/utils'
import { TrendingUp, TrendingDown } from 'lucide-react'

interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  trend?: {
    value: number
    isPositive: boolean
  }
  description?: string
  className?: string
  variant?: 'default' | 'success' | 'warning' | 'danger'
}

export function StatCard({
  title,
  value,
  icon,
  trend,
  description,
  className,
  variant = 'default',
}: StatCardProps) {
  const variants = {
    default: 'bg-white border-blue-100 hover:border-blue-200',
    success: 'bg-green-50 border-green-100 hover:border-green-200',
    warning: 'bg-yellow-50 border-yellow-100 hover:border-yellow-200',
    danger: 'bg-red-50 border-red-100 hover:border-red-200',
  }

  return (
    <motion.div
      whileHover={{ y: -5 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        'rounded-xl border p-6 shadow-sm transition-colors',
        variants[variant],
        className
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-500">{title}</span>
        <div className="rounded-full p-2 bg-opacity-10">
          {icon}
        </div>
      </div>

      <div className="mt-4">
        <h3 className="text-2xl font-bold">{value}</h3>
        {trend && (
          <div className="flex items-center mt-2">
            {trend.isPositive ? (
              <TrendingUp className="w-4 h-4 text-green-500" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-500" />
            )}
            <span
              className={cn(
                'text-sm ml-1',
                trend.isPositive ? 'text-green-600' : 'text-red-600'
              )}
            >
              {trend.value}%
            </span>
          </div>
        )}
        {description && (
          <p className="mt-2 text-sm text-gray-500">{description}</p>
        )}
      </div>
    </motion.div>
  )
}