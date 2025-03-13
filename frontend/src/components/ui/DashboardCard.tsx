'use client'

import { LucideProps, LucideIcon as Icon } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DashboardCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
  title?: string
  icon?: Icon
  className?: string
}

export function DashboardCard({ 
  children, 
  title, 
  icon: Icon, 
  className,
  ...props 
}: DashboardCardProps) {
  if (!Icon && !title) {
    return (
      <div 
        className={cn(
          "bg-white dark:bg-gray-800 rounded-lg shadow p-6",
          className
        )}
        {...props}
      >
        {children}
      </div>
    )
  }

  return (
    <div 
      className={cn(
        "bg-white dark:bg-gray-800 rounded-lg shadow p-6",
        className
      )}
      {...props}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          {Icon && (
            <div className="flex-shrink-0">
              <Icon className="h-6 w-6 text-blue-500" />
            </div>
          )}
          {title && (
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              {title}
            </h2>
          )}
        </div>
      </div>
      {children}
    </div>
  )
} 