import { useState, useEffect } from 'react'
import api from '../lib/api'
import { API_ENDPOINTS } from '../lib/api-endpoints'

interface AlertFilters {
  severity?: 'critical' | 'high' | 'medium' | 'low';
  status?: 'active' | 'resolved' | 'investigating';
  startDate?: string;
  endDate?: string;
  source?: string;
  page?: number;
  limit?: number;
}

interface Alert {
  id: string;
  title: string;
  description: string;
  severity: string;
  status: string;
  source: string;
  timestamp: string;
  assignedTo?: string;
  resolution?: string;
}

export function useAlerts(filters?: AlertFilters) {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [totalCount, setTotalCount] = useState(0)
  const [statistics, setStatistics] = useState<any>(null)

  const fetchAlerts = async () => {
    try {
      setLoading(true)
      const response = await api.get(API_ENDPOINTS.ALERTS.LIST, {
        params: filters
      })
      setAlerts(response.data.alerts)
      setTotalCount(response.data.total)
      setError(null)
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const fetchStatistics = async () => {
    try {
      const response = await api.get(API_ENDPOINTS.ALERTS.STATISTICS)
      setStatistics(response.data)
    } catch (err) {
      console.error('Failed to fetch alert statistics:', err)
    }
  }

  const updateAlert = async (alertId: string, data: Partial<Alert>) => {
    try {
      setLoading(true)
      await api.patch(API_ENDPOINTS.ALERTS.UPDATE(alertId), data)
      await fetchAlerts()
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const bulkUpdateAlerts = async (alertIds: string[], data: Partial<Alert>) => {
    try {
      setLoading(true)
      await api.post(API_ENDPOINTS.ALERTS.BULK_UPDATE, {
        alertIds,
        ...data
      })
      await fetchAlerts()
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const deleteAlert = async (alertId: string) => {
    try {
      setLoading(true)
      await api.delete(API_ENDPOINTS.ALERTS.DELETE(alertId))
      await fetchAlerts()
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAlerts()
    fetchStatistics()
  }, [filters])

  return {
    alerts,
    loading,
    error,
    totalCount,
    statistics,
    updateAlert,
    bulkUpdateAlerts,
    deleteAlert,
    refreshAlerts: fetchAlerts
  }
}