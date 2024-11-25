import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "../../lib/utils"
import React from 'react'

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none",
  {
    variants: {
      variant: {
        primary: "bg-slate-900 text-white hover:bg-slate-700",
        secondary: "bg-slate-100 text-slate-900 hover:bg-slate-200",
        outline: "border border-slate-200 hover:bg-slate-100",
        ghost: "hover:bg-slate-100 hover:text-slate-900",
        destructive: "bg-red-500 text-white hover:bg-red-600",
        link: "underline-offset-4 hover:underline text-slate-900",
      },
      size: {
        sm: "h-9 px-3 text-sm",
        md: "h-10 py-2 px-4",
        lg: "h-11 px-8",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { buttonVariants }