// src/app/dashboard/overview/page.tsx
import { Suspense } from 'react'
import { Card } from '@/components/ui/card'
import { AlertSummary } from './components/alert-summary'
import { SecurityMetrics } from './components/security-metrics'
import { RecentActivity } from './components/recent-activity'
import { SystemHealth } from './components/system-health'
import { LoadingSkeleton } from '@/components/ui/loading-skeleton'

export default function DashboardOverview() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">
        Security Overview
      </h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Suspense fallback={<LoadingSkeleton />}>
          <AlertSummary />
        </Suspense>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card className="p-6">
          <Suspense fallback={<LoadingSkeleton />}>
            <SecurityMetrics />
          </Suspense>
        </Card>
        
        <Card className="p-6">
          <Suspense fallback={<LoadingSkeleton />}>
            <SystemHealth />
          </Suspense>
        </Card>
      </div>

      <Card className="p-6">
        <Suspense fallback={<LoadingSkeleton />}>
          <RecentActivity />
        </Suspense>
      </Card>
    </div>
  )
}