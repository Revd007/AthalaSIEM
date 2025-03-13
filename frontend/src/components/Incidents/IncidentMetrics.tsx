'use client'

import { DashboardCard } from '@/components/ui/DashboardCard'
import { Clock, AlertTriangle, Shield, Activity } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'

export function IncidentMetrics() {
  const metrics = [
    {
      title: 'Active Incidents',
      value: '24',
      change: '+12%',
      icon: AlertTriangle,
      trend: 'up' as const,
      color: 'red' as const
    },
    {
      title: 'Avg. Response Time',
      value: '15m',
      change: '-8%',
      icon: Clock,
      trend: 'down' as const,
      color: 'green' as const
    },
    {
      title: 'Resolution Rate',
      value: '94%',
      change: '+5%',
      icon: Shield,
      trend: 'up' as const,
      color: 'blue' as const
    },
    {
      title: 'Critical Issues',
      value: '3',
      change: '-25%',
      icon: Activity,
      trend: 'down' as const,
      color: 'yellow' as const
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric) => (
        <StatsCard key={metric.title} {...metric} />
      ))}
    </div>
  )
} 