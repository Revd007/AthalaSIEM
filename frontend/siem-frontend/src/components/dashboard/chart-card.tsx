import React from 'react'
import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card'
import { ResponsiveContainer } from 'recharts'

interface ChartCardProps {
  title: string
  subtitle?: string
  chart: React.ReactElement
  filters?: React.ReactNode
  className?: string
}

export function ChartCard({
  title,
  subtitle,
  chart,
  filters,
  className,
}: ChartCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={className}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle className="text-base font-medium">{title}</CardTitle>
            {subtitle && (
              <p className="text-sm text-gray-500">{subtitle}</p>
            )}
          </div>
          {filters && <div className="flex items-center">{filters}</div>}
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              {chart}
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}