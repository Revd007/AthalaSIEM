'use client'

import { CheckCircle, AlertTriangle, Clock, Calendar } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import type { ComplianceFramework } from '@/types/compliance'

interface ComplianceMetricsProps {
  framework: ComplianceFramework
}

export function ComplianceMetrics({ framework }: ComplianceMetricsProps) {
  const metrics = [
    {
      title: 'Overall Compliance',
      value: '85%',
      change: '+5%',
      icon: CheckCircle,
      trend: 'up' as const,
      color: 'green' as const
    },
    {
      title: 'Controls at Risk',
      value: '12',
      change: '-2',
      icon: AlertTriangle,
      trend: 'down' as const,
      color: 'red' as const
    },
    {
      title: 'Pending Reviews',
      value: '8',
      change: '+3',
      icon: Clock,
      trend: 'up' as const,
      color: 'yellow' as const
    },
    {
      title: 'Next Audit',
      value: '45 days',
      change: '',
      icon: Calendar,
      trend: 'neutral' as const,
      color: 'blue' as const
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