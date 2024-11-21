import React from 'react'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Card({ children, className, ...props }: CardProps) {
  return (
    <div className={`rounded-lg border border-gray-200 bg-white ${className || ''}`} {...props}>
      {children}
    </div>
  )
}