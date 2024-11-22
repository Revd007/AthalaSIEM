import React from 'react'
import { cn } from '../../../lib/utils'

interface TextInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  icon?: React.ComponentType<{ className?: string }>
  rightElement?: React.ReactNode
}

export const TextInput = React.forwardRef<HTMLInputElement, TextInputProps>(
  ({ className, label, error, icon: Icon, rightElement, ...props }, ref) => {
    return (
      <div className="space-y-1">
        {label && (
          <label className="text-sm font-medium text-gray-700">{label}</label>
        )}
        <div className="relative">
          {Icon && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Icon className="h-5 w-5 text-gray-400" />
            </div>
          )}
          <input
            ref={ref}
            className={cn(
              'block w-full rounded-md border border-gray-300 shadow-sm',
              'focus:border-primary-500 focus:ring-1 focus:ring-primary-500',
              'disabled:bg-gray-50 disabled:text-gray-500 disabled:border-gray-200',
              'sm:text-sm',
              Icon && 'pl-10',
              rightElement && 'pr-10',
              error && 'border-red-300 focus:border-red-500 focus:ring-red-500',
              className
            )}
            {...props}
          />
          {rightElement && (
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
              {rightElement}
            </div>
          )}
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    )
  }
)