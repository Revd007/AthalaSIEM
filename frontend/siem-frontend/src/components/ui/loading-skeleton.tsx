import React from 'react';

interface LoadingSkeletonProps {
  rows?: number
  height?: string
  className?: string
}

export function LoadingSkeleton({ 
  rows = 3, 
  height = "h-4",
  className = "" 
}: LoadingSkeletonProps) {
  return (
    <div className={`animate-pulse space-y-4 ${className}`}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className={`${height} bg-gray-200 rounded ${
            i === 0 ? 'w-3/4' : i === rows - 1 ? 'w-5/6' : 'w-full'
          }`}
        />
      ))}
    </div>
  )
}