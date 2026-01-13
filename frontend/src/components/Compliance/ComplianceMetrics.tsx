'use client'

import { CheckCircle, AlertTriangle, Clock, Calendar } from 'lucide-react'
import { StatsCard } from '@/components/ui/StatsCard'
import { useQuery } from '@tanstack/react-query'
import { complianceService } from '@/services/compliance-service'
import { Skeleton } from '@/components/ui/skeleton'
import type { ComplianceFramework } from '@/types/compliance'

interface ComplianceMetricsProps {
  framework: ComplianceFramework
}

export function ComplianceMetrics({ framework }: ComplianceMetricsProps) {
  const { data: metricsData, isLoading } = useQuery({
    queryKey: ['compliance-metrics', framework],
    queryFn: () => complianceService.getMetrics(framework),
    refetchInterval: 300000, // 5 minutes
  });

  const metrics = [
    {
      title: 'Overall Compliance',
      value: metricsData ? `${metricsData.overallCompliance}%` : 'N/A',
      change: '+0%',
      icon: CheckCircle,
      trend: 'up' as const,
      color: 'green' as const
    },
    {
      title: 'Controls at Risk',
      value: metricsData?.controlsAtRisk?.toString() || '0',
      change: '+0',
      icon: AlertTriangle,
      trend: 'down' as const,
      color: 'red' as const
    },
    {
      title: 'Pending Reviews',
      value: metricsData?.pendingReviews?.toString() || '0',
      change: '+0',
      icon: Clock,
      trend: 'up' as const,
      color: 'yellow' as const
    },
    {
      title: 'Next Audit',
      value: metricsData?.nextAuditDate 
        ? `${Math.ceil((new Date(metricsData.nextAuditDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))} days`
        : 'N/A',
      change: '',
      icon: Calendar,
      trend: 'neutral' as const,
      color: 'blue' as const
    }
  ]

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-32 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric) => (
        <StatsCard key={metric.title} {...metric} />
      ))}
    </div>
  )
} 