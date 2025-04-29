'use client'

import { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DashboardCardProps {
  title: string
  icon: LucideIcon
  children: React.ReactNode
  className?: string
}

export function DashboardCard({ title, icon: Icon, children, className }: DashboardCardProps) {
  return (
    <div className={cn(
      "bg-white dark:bg-gray-800 rounded-lg shadow p-6",
      className
    )}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">{title}</h3>
        <Icon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
      </div>
      {children}
    </div>
  )
} 