import { ReactNode } from 'react'

interface StatCardProps {
  title: string
  value: string
  icon?: ReactNode
  description?: string
  variant?: 'default' | 'danger'
}

export function StatCard({ 
  title, 
  value, 
  icon, 
  description, 
  variant = 'default' 
}: StatCardProps) {
  return (
    <div className={`rounded-lg border bg-white p-6 ${
      variant === 'danger' ? 'border-red-200 bg-red-50' : ''
    }`}>
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-semibold">{value}</p>
          {description && (
            <p className="text-sm text-gray-500">{description}</p>
          )}
        </div>
        {icon && <div>{icon}</div>}
      </div>
    </div>
  )
}