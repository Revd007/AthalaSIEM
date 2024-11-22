import { useState, useEffect } from 'react'
import api from '../lib/api'
import { API_ENDPOINTS } from '../lib/api-endpoints'

interface LogFilters {
  startDate?: string;
  endDate?: string;
  source?: string;
  type?: string;
  severity?: string;
  search?: string;
  page?: number;
  limit?: number;
}

export function useLogs(filters?: LogFilters) {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [total, setTotal] = useState(0)

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        setLoading(true)
        const response = await api.get(API_ENDPOINTS.LOGS.LIST, {
          params: filters
        })
        setLogs(response.data.logs)
        setTotal(response.data.total)
        setError(null)
      } catch (err) {
        setError(err as Error)
      } finally {
        setLoading(false)
      }
    }

    fetchLogs()
  }, [filters])

  const searchLogs = async (searchTerm: string) => {
    try {
      setLoading(true)
      const response = await api.post(API_ENDPOINTS.LOGS.SEARCH, {
        searchTerm
      })
      setLogs(response.data.logs)
      setTotal(response.data.total)
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const exportLogs = async (format: 'csv' | 'pdf') => {
    try {
      const response = await api.get(API_ENDPOINTS.LOGS.EXPORT, {
        params: { ...filters, format },
        responseType: 'blob'
      })
      return response.data
    } catch (err) {
      setError(err as Error)
      return null
    }
  }

  return { logs, loading, error, total, searchLogs, exportLogs }
}