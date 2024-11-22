import { useState, useEffect } from 'react'
import api from '../lib/api'
import { API_ENDPOINTS } from '../lib/api-endpoints'

interface DashboardWidget {
  id: string;
  type: string;
  title: string;
  data: any;
  settings: any;
  position: { x: number; y: number };
}

interface DashboardData {
  widgets: DashboardWidget[];
  layout: any;
  settings: any;
}

export function useDashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const fetchDashboard = async () => {
    try {
      setLoading(true)
      const [overview, widgets] = await Promise.all([
        api.get(API_ENDPOINTS.DASHBOARD.OVERVIEW),
        api.get(API_ENDPOINTS.DASHBOARD.WIDGETS)
      ])
      
      setDashboardData({
        ...overview.data,
        widgets: widgets.data
      })
      setError(null)
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const updateWidgetSettings = async (widgetId: string, settings: any) => {
    try {
      setLoading(true)
      await api.patch(API_ENDPOINTS.DASHBOARD.WIDGETS, {
        widgetId,
        settings
      })
      await fetchDashboard()
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const saveLayout = async (layout: any) => {
    try {
      setLoading(true)
      await api.post(API_ENDPOINTS.DASHBOARD.CUSTOMIZE, { layout })
      await fetchDashboard()
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDashboard()
  }, [])

  return {
    dashboardData,
    loading,
    error,
    updateWidgetSettings,
    saveLayout,
    refreshDashboard: fetchDashboard
  }
}