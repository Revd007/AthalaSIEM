import { useState } from 'react'
import api from '../lib/api'
import { API_ENDPOINTS } from '../lib/api-endpoints'

interface ReportConfig {
  template: string;
  startDate: string;
  endDate: string;
  format: 'pdf' | 'csv';
  includeCharts: boolean;
}

export function useReports() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const generateReport = async (config: ReportConfig) => {
    try {
      setLoading(true)
      const response = await api.post(API_ENDPOINTS.REPORTS.GENERATE, config)
      return response.data
    } catch (err) {
      setError(err as Error)
      return null
    } finally {
      setLoading(false)
    }
  }

  const downloadReport = async (reportId: string) => {
    try {
      setLoading(true)
      const response = await api.get(
        API_ENDPOINTS.REPORTS.DOWNLOAD(reportId),
        { responseType: 'blob' }
      )
      return response.data
    } catch (err) {
      setError(err as Error)
      return null
    } finally {
      setLoading(false)
    }
  }

  return { generateReport, downloadReport, loading, error }
}