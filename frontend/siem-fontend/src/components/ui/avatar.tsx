import React from 'react'
import { cn } from '../../lib/utils'

interface AvatarProps {
  src?: string
  fallback: string
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function Avatar({ src, fallback, size = 'md', className }: AvatarProps) {
  const sizeClasses = {
    sm: 'w-8 h-8 text-xs',
    md: 'w-10 h-10 text-sm',
    lg: 'w-12 h-12 text-base',
  }

  return (
    <div
      className={cn(
        'relative rounded-full overflow-hidden bg-gray-100',
        sizeClasses[size],
        className
      )}
    >
      {src ? (
        <img
          src={src}
          alt="Avatar"
          className="w-full h-full object-cover"
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-primary-100 text-primary-600 font-medium">
          {fallback}
        </div>
      )}
    </div>
  )
}