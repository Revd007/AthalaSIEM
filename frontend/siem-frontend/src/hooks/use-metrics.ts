import { useState, useEffect } from 'react'
import api from '../lib/api'
import { API_ENDPOINTS } from '../lib/api-endpoints'

export function useMetrics() {
  const [metrics, setMetrics] = useState({
    summary: null,
    trends: null,
    securityScore: null,
    systemHealth: null
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const fetchAllMetrics = async () => {
    try {
      setLoading(true)
      const [summary, trends, securityScore, systemHealth] = await Promise.all([
        api.get(API_ENDPOINTS.METRICS.SUMMARY),
        api.get(API_ENDPOINTS.METRICS.TRENDS),
        api.get(API_ENDPOINTS.METRICS.SECURITY_SCORE),
        api.get(API_ENDPOINTS.METRICS.SYSTEM_HEALTH)
      ])

      setMetrics({
        summary: summary.data,
        trends: trends.data,
        securityScore: securityScore.data,
        systemHealth: systemHealth.data
      })
      setError(null)
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAllMetrics()
  }, [])

  return { metrics, loading, error, refreshMetrics: fetchAllMetrics }
}