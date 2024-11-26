'use client'

import { useEffect, useState } from 'react'
import { useAuthStore } from '../../hooks/use-auth'
import { useRouter } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'

interface DashboardStats {
  totalAlerts: number;
  criticalAlerts: number;
  activeEvents: number;
  systemHealth: number;
}

interface RecentAlert {
  id: string;
  title: string;
  severity: string;
  timestamp: string;
}

// API fetch functions
const fetchStats = async (): Promise<DashboardStats> => {
  const response = await fetch('/api/dashboard/stats', {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('token')}`
    }
  })
  if (!response.ok) throw new Error('Failed to fetch stats')
  return response.json()
}

const fetchRecentAlerts = async (): Promise<RecentAlert[]> => {
  const response = await fetch('/api/dashboard/recent-alerts', {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem('token')}`
    }
  })
  if (!response.ok) throw new Error('Failed to fetch alerts')
  return response.json()
}

export default function Dashboard() {
  const { user } = useAuthStore()
  const router = useRouter()

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboardStats'],
    queryFn: fetchStats,
    initialData: {
      totalAlerts: 0,
      criticalAlerts: 0,
      activeEvents: 0,
      systemHealth: 0
    }
  })

  const { data: recentAlerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['recentAlerts'],
    queryFn: fetchRecentAlerts,
    initialData: []
  })

  if (statsLoading || alertsLoading) {
    return <div>Loading...</div>
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Welcome back, {user?.username}</h1>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          title="Total Alerts"
          value={stats.totalAlerts}
          icon="🚨"
          color="bg-blue-500"
        />
        <StatCard
          title="Critical Alerts"
          value={stats.criticalAlerts}
          icon="⚠️"
          color="bg-red-500"
        />
        <StatCard
          title="Active Events"
          value={stats.activeEvents}
          icon="📊"
          color="bg-green-500"
        />
        <StatCard
          title="System Health"
          value={`${stats.systemHealth}%`}
          icon="💪"
          color="bg-purple-500"
        />
      </div>

      {/* Recent Alerts */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Alerts</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2">Title</th>
                <th className="text-left py-2">Severity</th>
                <th className="text-left py-2">Time</th>
              </tr>
            </thead>
            <tbody>
              {recentAlerts.map((alert) => (
                <tr key={alert.id} className="border-b hover:bg-gray-50">
                  <td className="py-2">{alert.title}</td>
                  <td className="py-2">
                    <SeverityBadge severity={alert.severity} />
                  </td>
                  <td className="py-2">{new Date(alert.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StatCard({ title, value, icon, color }: { 
  title: string; 
  value: number | string; 
  icon: string;
  color: string;
}) {
  return (
    <div className={`${color} text-white rounded-lg p-6 shadow-lg`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm opacity-80">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
        </div>
        <span className="text-2xl">{icon}</span>
      </div>
    </div>
  )
}

function SeverityBadge({ severity }: { severity: string }) {
  const colors = {
    critical: 'bg-red-100 text-red-800',
    high: 'bg-orange-100 text-orange-800',
    medium: 'bg-yellow-100 text-yellow-800',
    low: 'bg-green-100 text-green-800'
  }

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[severity.toLowerCase()]}`}>
      {severity}
    </span>
  )
}