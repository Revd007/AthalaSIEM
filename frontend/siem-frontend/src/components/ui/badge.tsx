import React from 'react';

export interface BadgeProps {
  variant?: 'default' | 'secondary' | 'destructive' | 'success' | 'warning' | 'info' | 'outline';
  children: React.ReactNode;
  className?: string;
}

export function Badge({ 
  variant = 'default', 
  children,
  className = ''
}: BadgeProps) {
  const baseStyles = "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2";
  
  const variants = {
    default: "bg-primary text-primary-foreground",
    secondary: "bg-secondary text-secondary-foreground",
    destructive: "bg-destructive text-destructive-foreground",
    success: "bg-green-100 text-green-800",
    warning: "bg-yellow-100 text-yellow-800",
    info: "bg-blue-100 text-blue-800",
    outline: "text-foreground border border-input",
  };

  return (
    <div className={`${baseStyles} ${variants[variant]} ${className}`}>
      {children}
    </div>
  );
}