'use client'

import { Suspense, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../../hooks/use-auth'
import { Card } from '../../../components/ui/card'
import { SystemHealth } from '../../../components/dashboard/components/system-health'
import { ThreatMap } from '../../../components/dashboard/components/threat-map'
import { SecurityAlerts } from '../../../components/dashboard/security-alerts'
import { SystemMonitor } from '../../../components/dashboard/system-monitor'
import { EventsOverview } from '../../../components/dashboard/components/events-overview'
import { AIInsights } from '../../../components/dashboard/ai-insights'
import { RecentActivity } from '../../../components/dashboard/recent-activity'

function LoadingFallback() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 bg-gray-200 rounded w-1/4"></div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="h-32 bg-gray-200 rounded"></div>
        <div className="h-32 bg-gray-200 rounded"></div>
        <div className="h-32 bg-gray-200 rounded"></div>
        <div className="h-32 bg-gray-200 rounded"></div>
      </div>
    </div>
  )
}

export default function DashboardOverview() {
  const router = useRouter()
  const { token, initialized } = useAuthStore()

  useEffect(() => {
    if (initialized && !token) {
      router.replace('/login')
    }
  }, [token, initialized, router])

  if (!initialized || !token) return null

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <Suspense fallback={<LoadingFallback />}>
        {/* Top Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Suspense fallback={<div className="h-32 bg-gray-200 rounded animate-pulse"></div>}>
            <SystemMonitor />
          </Suspense>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            <Card className="p-6">
              <SecurityAlerts />
            </Card>
            
            <Card className="p-6">
              <AIInsights />
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            <Card className="p-6">
              <ThreatMap />
            </Card>
            
            <Card className="p-6">
              <SystemHealth />
            </Card>
            
            <Card className="p-6">
              <EventsOverview />
            </Card>
          </div>
        </div>

        {/* Bottom Section */}
        <Card className="p-6">
          <RecentActivity />
        </Card>
      </Suspense>
    </div>
  )
}