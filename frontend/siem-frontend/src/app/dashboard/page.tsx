'use client'

import { useEffect, useState } from 'react'
import DashboardLayout from '../../components/layouts/DashboardLayout'
import { useQuery } from '@tanstack/react-query'
import { StatCard } from '../../components/dashboard/stat-card'
import { SeverityBadge } from '../../components/dashboard/severity-badge'
interface DashboardStats {
  totalAlerts: number
  activeThreats: number
  systemHealth: number
  recentEvents: Array<{
    id: string
    type: string
    severity: string
    timestamp: string
    description: string
  }>
}

async function fetchDashboardStats(): Promise<DashboardStats> {
  const response = await fetch('/api/dashboard/stats', {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('token')}`
    }
  })
  
  if (!response.ok) {
    throw new Error('Failed to fetch dashboard stats')
  }
  
  return response.json()
}

export default function Dashboard() {
  const { data: stats, isLoading, error } = useQuery({
    queryKey: ['dashboardStats'],
    queryFn: fetchDashboardStats
  })

  if (error) {
    return (
      <DashboardLayout>
        <div className="text-red-500">Error loading dashboard data</div>
      </DashboardLayout>
    )
  }

  if (isLoading) {
    return (
      <DashboardLayout>
        <div>Loading dashboard data...</div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          title="Total Alerts"
          value={stats?.totalAlerts || 0}
          icon="🚨"
        />
        <StatCard
          title="Active Threats"
          value={stats?.activeThreats || 0}
          icon="⚠️"
        />
        <StatCard
          title="System Health"
          value={`${stats?.systemHealth || 0}%`}
          icon="💻"
        />
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Events</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="px-6 py-3 border-b">Type</th>
                <th className="px-6 py-3 border-b">Severity</th>
                <th className="px-6 py-3 border-b">Timestamp</th>
                <th className="px-6 py-3 border-b">Description</th>
              </tr>
            </thead>
            <tbody>
              {stats?.recentEvents.map((event) => (
                <tr key={event.id}>
                  <td className="px-6 py-4 border-b">{event.type}</td>
                  <td className="px-6 py-4 border-b">
                    <SeverityBadge severity={event.severity} />
                  </td>
                  <td className="px-6 py-4 border-b">
                    {new Date(event.timestamp).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 border-b">{event.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </DashboardLayout>
  )
}

// ... StatCard and SeverityBadge components remain the same ...