import React from 'react'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export function Input({ className, ...props }: InputProps) {
  return (
    <input
      className={`rounded-md border border-gray-300 px-3 py-2 ${className || ''}`}
      {...props}
    />
  )
}