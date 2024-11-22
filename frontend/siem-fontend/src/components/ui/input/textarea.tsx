import React from 'react'
import { cn } from '../../../lib/utils'

interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, ...props }, ref) => {
    return (
      <div className="space-y-1">
        {label && (
          <label className="text-sm font-medium text-gray-700">{label}</label>
        )}
        <textarea
          ref={ref}
          className={cn(
            'block w-full rounded-md border border-gray-300 shadow-sm',
            'focus:border-primary-500 focus:ring-1 focus:ring-primary-500',
            'disabled:bg-gray-50 disabled:text-gray-500 disabled:border-gray-200',
            'sm:text-sm min-h-[80px]',
            error && 'border-red-300 focus:border-red-500 focus:ring-red-500',
            className
          )}
          {...props}
        />
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    )
  }
)