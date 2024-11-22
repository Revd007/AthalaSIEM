import React from 'react'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Card({ children, className, ...props }: CardProps) {
  return (
    <div className={`rounded-lg border border-gray-200 bg-white ${className || ''}`} {...props}>
      {children}
    </div>
  )
}

export function CardHeader({ children, className, ...props }: CardProps) {
  return (
    <div className={`flex flex-col space-y-1.5 p-6 ${className || ''}`} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({ children, className, ...props }: CardProps) {
  return (
    <h3 className={`text-lg font-semibold leading-none tracking-tight ${className || ''}`} {...props}>
      {children}
    </h3>
  )
}

export function CardContent({ children, className, ...props }: CardProps) {
  return (
    <div className={`p-6 pt-0 ${className || ''}`} {...props}>
      {children}
    </div>
  )
}