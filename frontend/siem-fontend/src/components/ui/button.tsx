import React from 'react'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'outline'
}

export function Button({ variant = 'primary', children, className, ...props }: ButtonProps) {
  const baseStyles = 'px-4 py-2 rounded-md'
  const variantStyles = variant === 'primary' 
    ? 'bg-blue-600 text-white hover:bg-blue-700' 
    : 'border border-gray-300 hover:bg-gray-50'

  return (
    <button className={`${baseStyles} ${variantStyles} ${className || ''}`} {...props}>
      {children}
    </button>
  )
}