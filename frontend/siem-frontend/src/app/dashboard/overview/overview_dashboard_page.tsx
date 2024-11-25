// src/app/dashboard/overview/page.tsx
'use client'

import { Suspense, useEffect, useState } from 'react'
import { Card } from '../../../components/ui/card'
import AlertSummary from '../../../components/dashboard/alert-summary'
import { RecentActivity } from '../../../components/dashboard/recent-activity'
import { SystemHealth } from '../../../components/dashboard/components/system-health'
import { LoadingSkeleton } from '../../../components/ui/loading-skeleton'
import { ThreatMap } from '../../../components/dashboard/components/threat-map'
import { AIInsights } from '../../../components/dashboard/ai-insights'
import { axiosInstance } from '../../../lib/axios'
import React from 'react'

// Define interfaces for type safety
interface DashboardData {
  summary: {
    alerts: {
      critical: number;
      high: number;
      medium: number;
      low: number;
      total: number;
    };
  };
  metrics: {
    events: number;
    threats: number;
    incidents: number;
    // Add other metric fields as needed
  };
  health: {
    name: string;
    status: 'healthy' | 'warning' | 'critical';
    uptime: string;
    components: {
      name: string;
      status: 'healthy' | 'warning' | 'critical';
      uptime: string;
    }[];
  };
}

export default function DashboardOverview() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const [summaryRes, metricsRes, healthRes] = await Promise.all([
          axiosInstance.get('/api/v1/dashboard/summary'),
          axiosInstance.get('/api/v1/dashboard/metrics'),
          axiosInstance.get('/api/v1/system/health')
        ])

        setDashboardData({
          summary: summaryRes.data,
          metrics: metricsRes.data,
          health: healthRes.data
        })
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchDashboardData()
    // Refresh every 5 minutes
    const interval = setInterval(fetchDashboardData, 300000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <LoadingSkeleton />

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-gray-900">
        Security Overview
      </h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Suspense fallback={<LoadingSkeleton />}>
          {dashboardData?.summary?.alerts && (
            <AlertSummary alerts={{
              critical: dashboardData.summary.alerts.critical,
              high: dashboardData.summary.alerts.high,
              medium: dashboardData.summary.alerts.medium,
              low: dashboardData.summary.alerts.low,
              total: dashboardData.summary.alerts.critical + 
                     dashboardData.summary.alerts.high + 
                     dashboardData.summary.alerts.medium + 
                     dashboardData.summary.alerts.low
            }} />
          )}
        </Suspense>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <Suspense fallback={<LoadingSkeleton />}>
            <div className="h-[300px]">
              {/* Replace SecurityMetrics with a chart component */}
              {dashboardData?.metrics && (
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Security Metrics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Total Events</p>
                      <p className="text-2xl font-bold">{dashboardData.metrics.events}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Active Threats</p>
                      <p className="text-2xl font-bold">{dashboardData.metrics.threats}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </Suspense>
        </Card>
        
        <Card className="p-6">
          <Suspense fallback={<LoadingSkeleton />}>
            {dashboardData?.health && (
              <SystemHealth healthData={{
                name: dashboardData.health.name,
                status: dashboardData.health.status,
                uptime: dashboardData.health.uptime,
                components: dashboardData.health.components
              }} />
            )}
          </Suspense>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <Suspense fallback={<LoadingSkeleton />}>
            <ThreatMap />
          </Suspense>
        </Card>
        
        <Card className="p-6">
          <Suspense fallback={<LoadingSkeleton />}>
            <AIInsights />
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