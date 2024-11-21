import { forwardRef } from 'react'

interface TableProps extends React.HTMLAttributes<HTMLTableElement> {}

export const Table = forwardRef<HTMLTableElement, TableProps>(
  ({ className, ...props }, ref) => (
    <table
      ref={ref}
      className={`min-w-full divide-y divide-gray-200 ${className}`}
      {...props}
    />
  )
)

Table.displayName = 'Table'