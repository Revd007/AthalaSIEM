import * as React from 'react'
import * as ProgressPrimitive from '@radix-ui/react-progress'
import { cn } from '../../lib/utils'

interface ProgressProps
  extends React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> {
  variant?: 'default' | 'success' | 'warning' | 'danger'
  showValue?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  ProgressProps
>(({ className, value, variant = 'default', showValue, size = 'md', ...props }, ref) => {
  const getVariantClass = () => {
    switch (variant) {
      case 'success':
        return 'bg-green-600'
      case 'warning':
        return 'bg-yellow-600'
      case 'danger':
        return 'bg-red-600'
      default:
        return 'bg-primary'
    }
  }

  const getSizeClass = () => {
    switch (size) {
      case 'sm':
        return 'h-2'
      case 'lg':
        return 'h-6'
      default:
        return 'h-4'
    }
  }

  return (
    <div className="relative">
      <ProgressPrimitive.Root
        ref={ref}
        className={cn(
          'relative w-full overflow-hidden rounded-full bg-gray-200',
          getSizeClass(),
          className
        )}
        {...props}
      >
        <ProgressPrimitive.Indicator
          className={cn(
            'h-full w-full flex-1 transition-all',
            getVariantClass()
          )}
          style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
        />
      </ProgressPrimitive.Root>
      {showValue && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-medium text-white">
            {Math.round(value || 0)}%
          </span>
        </div>
      )}
    </div>
  )
})

Progress.displayName = ProgressPrimitive.Root.displayName

export { Progress }