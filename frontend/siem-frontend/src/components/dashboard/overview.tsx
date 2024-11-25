'use client'

import { useEffect, useState } from 'react'
import { Card } from '../ui/card'
import AlertSummary from './alert-summary'
import { SystemHealth } from './components/system-health'
import { ThreatMap } from './components/threat-map'
import { AIInsights } from './ai-insights'
import { RecentActivity } from './recent-activity'
import { EventTrendChart } from './charts/event-trend-chart'
import { useDashboardData } from '../../hooks/use-dashboard-data'
import { LoadingSkeleton } from '../ui/loading-skeleton'

export function DashboardOverview() {
  const { data, loading } = useDashboardData()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || loading) return <LoadingSkeleton />

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-gray-900">
        Security Overview
      </h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <AlertSummary alerts={data?.summary?.alerts} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <EventTrendChart data={data?.metrics?.eventTrend || []} />
        </Card>
        
        <Card className="p-6">
          <SystemHealth healt
           
           hData={data?.health} />
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <ThreatMap />
        </Card>
        
        <Card className="p-6">
          <AIInsights />
        </Card>
      </div>

      <Card className="p-6">
        <RecentActivity />
      </Card>
    </div>
  )
}