'use client'

import { Server, Shield, Network, Monitor, Database, AlertTriangle } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'

export function SystemHealthOverview() {
  const metrics = [
    {
      title: 'Total Devices',
      value: '156',
      change: '+3',
      icon: Server,
      trend: 'up' as const,
      color: 'blue' as const
    },
    {
      title: 'Healthy',
      value: '142',
      change: '+5',
      icon: Shield,
      trend: 'up' as const,
      color: 'green' as const
    },
    {
      title: 'Warning',
      value: '8',
      change: '-2',
      icon: AlertTriangle,
      trend: 'down' as const,
      color: 'yellow' as const
    },
    {
      title: 'Critical',
      value: '6',
      change: '+1',
      icon: AlertTriangle,
      trend: 'up' as const,
      color: 'red' as const
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