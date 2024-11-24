'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '../../../hooks/use-auth'
import { Card } from '../../../components/ui/card'
import { SecurityMetrics } from '../../../components/dashboard/SecurityMetrics'
import { SystemHealth } from '../../../components/dashboard/components/system-health'
import { ThreatMap } from '../../../components/dashboard/components/threat-map'
import { SecurityAlerts } from '../../../components/dashboard/security-alerts'
import { SystemMonitor } from '../../../components/dashboard/system-monitor'
import { EventsOverview } from '../../../components/dashboard/components/events-overview'
import { AIInsights } from '../../../components/dashboard/ai-insights'
import { RecentActivity } from '../../../components/dashboard/recent-activity'

export default function DashboardOverview() {
  const router = useRouter()
  const { token } = useAuthStore()

  useEffect(() => {
    if (!token) {
      router.push('/login')
    }
  }, [token, router])

  if (!token) return null

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <SystemMonitor />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          <Card className="p-6">
            <SecurityAlerts />
          </Card>
          
          <Card className="p-6">
            <SecurityMetrics />
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
    </div>
  )
}