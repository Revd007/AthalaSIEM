import { forwardRef } from 'react'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={`rounded-lg border bg-white text-gray-950 shadow ${className}`}
      {...props}
    />
  )
)

Card.displayName = 'Card'