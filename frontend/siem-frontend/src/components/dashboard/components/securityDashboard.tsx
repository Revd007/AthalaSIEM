'use client'

import React, { useState, useEffect } from 'react'
import AlertSummary from '../alert-summary'
import { SystemHealth } from './system-health'
import { ThreatMap } from './threat-map'
import { Card } from '../../ui/card'
import { SecurityMetrics } from '../SecurityMetrics'

// Define interfaces
interface ThreatAnalysis {
  risk_level: string
  recommendations: string[]
  details: {
    source: string
    type: string
    severity: string
  }
}

interface AnomalyResult {
  detected: boolean
  score: number
  details: string
}

interface SystemMetrics {
  cpu_usage: number
  memory_usage: number
  disk_usage: number
  network_traffic: number
}

export const SecurityDashboard: React.FC = () => {
  const [mounted, setMounted] = useState(false)
  const [threatAnalysis, setThreatAnalysis] = useState<ThreatAnalysis | null>(null)
  const [anomalyData, setAnomalyData] = useState<AnomalyResult | null>(null)
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null)

  useEffect(() => {
    setMounted(true)
    const fetchData = async () => {
      try {
        // Fetch system metrics
        const metricsResponse = await fetch('/api/metrics')
        const metrics = await metricsResponse.json()
        setSystemMetrics(metrics)

        // Fetch threat analysis
        const threatResponse = await fetch('/api/threats/analysis')
        const threatData = await threatResponse.json()
        setThreatAnalysis(threatData)

        // Fetch anomaly data
        const anomalyResponse = await fetch('/api/anomalies')
        const anomalyResult = await anomalyResponse.json()
        setAnomalyData(anomalyResult)
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
      }
    }

    if (mounted) {
      fetchData()
      const interval = setInterval(fetchData, 300000) // Update every 5 minutes
      return () => clearInterval(interval)
    }
  }, [mounted])

  if (!mounted) return null

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card className="p-6">
        <SecurityMetrics />
      </Card>
      
      <Card className="p-6">
        <AlertSummary data={threatAnalysis} />
      </Card>
      
      <Card className="p-6">
        <SystemHealth metrics={systemMetrics} />
      </Card>
      
      <Card className="p-6">
        <ThreatMap />
      </Card>
    </div>
  )
}

export default SecurityDashboard