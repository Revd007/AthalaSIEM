import { useEffect, useState } from 'react'
import { axiosInstance } from '../lib/axios'

interface DashboardData {
  summary: {
    alerts: {
      critical: number
      high: number
      medium: number
      low: number
    }
  }
  metrics: {
    events: number
    threats: number
    incidents: number
    eventTrend: any[]
  }
  health: {
    status: string
    components: {
      name: string
      status: 'healthy' | 'warning' | 'critical'
      uptime: string
    }[]
  }
}

export function useDashboardData() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const [summaryRes, metricsRes, healthRes] = await Promise.all([
          axiosInstance.get('/api/v1/dashboard/summary'),
          axiosInstance.get('/api/v1/dashboard/metrics'),
          axiosInstance.get('/api/v1/system/health')
        ])

        setData({
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
    const interval = setInterval(fetchDashboardData, 300000)
    return () => clearInterval(interval)
  }, [])

  return { data, loading }
}