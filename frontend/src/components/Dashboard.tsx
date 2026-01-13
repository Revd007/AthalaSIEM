'use client'

import { Activity, AlertCircle, Shield, TrendingUp } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Skeleton } from '@/components/ui/skeleton'

interface DashboardStats {
  totalEvents: number
  criticalAlerts: number
  resolvedIncidents: number
  eventsPerSecond: number
}

export function Dashboard() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      try {
        const response = await api.get<DashboardStats>('/api/dashboard/metrics')
        return response.data
      } catch {
        // Return default values if API fails
        return {
          totalEvents: 0,
          criticalAlerts: 0,
          resolvedIncidents: 0,
          eventsPerSecond: 0
        }
      }
    },
    staleTime: 30000,
  })

  const statItems = [
    { 
      label: 'Total Events', 
      value: stats?.totalEvents?.toLocaleString() || '0', 
      icon: Activity, 
      change: '+0%',
      color: 'blue'
    },
    { 
      label: 'Critical Alerts', 
      value: stats?.criticalAlerts?.toString() || '0', 
      icon: AlertCircle, 
      change: '+0%',
      color: 'red'
    },
    { 
      label: 'Resolved', 
      value: stats?.resolvedIncidents?.toString() || '0', 
      icon: Shield, 
      change: '+0%',
      color: 'green'
    },
    { 
      label: 'Events/sec', 
      value: stats?.eventsPerSecond?.toFixed(1) || '0', 
      icon: TrendingUp, 
      change: '+0%',
      color: 'purple'
    },
  ]

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-28 w-full" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {statItems.map((stat, index) => {
        const Icon = stat.icon
        const colorClasses = {
          blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
          red: 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400',
          green: 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400',
          purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
        }
        
        return (
          <div 
            key={index}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm"
          >
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">{stat.label}</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
                  {stat.value}
                </p>
                <span className="text-sm text-green-500">{stat.change}</span>
              </div>
              <div className={`p-3 rounded-lg ${colorClasses[stat.color as keyof typeof colorClasses]}`}>
                <Icon className="h-6 w-6" />
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}