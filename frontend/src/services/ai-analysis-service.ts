import { api } from '@/lib/api'
import type {
  AIAnalysisOverview,
  AIAnomaly,
  AIPrediction,
  PlaybookExecution,
} from '@/types/api'

export const aiAnalysisService = {
  /** Dashboard stats from Python backend */
  async getOverview(): Promise<AIAnalysisOverview> {
    const { data } = await api.get<AIAnalysisOverview>('/api/ai-analysis/overview')
    return data ?? {
      activeThreats: 0,
      avgConfidence: 0,
      detectionRate24h: 0,
      responseTime: '-',
      mitreCoveragePercent: 0,
      insightsTrend: [],
      latestInsights: [],
    }
  },

  /** Anomaly detection data (counts + hourly + list) */
  async getAnomalies(): Promise<{
    total: number
    highSeverity: number
    detectedToday: number
    hourly: Array<{ time: string; count: number }>
    anomalies: AIAnomaly[]
  }> {
    const { data } = await api.get<{
      total: number
      highSeverity: number
      detectedToday: number
      hourly: Array<{ time: string; count: number }>
      anomalies: AIAnomaly[]
    }>('/api/ai-analysis/anomalies')
    return (
      data ?? {
        total: 0,
        highSeverity: 0,
        detectedToday: 0,
        hourly: [],
        anomalies: [],
      }
    )
  },

  /** User/behavior analysis */
  async getBehavior(): Promise<{
    topBehaviors: Array<{ name: string; count: number }>
    riskScore: number
    anomalies: unknown[]
  }> {
    const { data } = await api.get<{
      topBehaviors: Array<{ name: string; count: number }>
      riskScore: number
      anomalies: unknown[]
    }>('/api/ai-analysis/behavior')
    return (
      data ?? {
        topBehaviors: [],
        riskScore: 0,
        anomalies: [],
      }
    )
  },

  /** Predictive analysis */
  async getPredictive(): Promise<{
    predictions: AIPrediction[]
    hourly: Array<{ time: string; count: number }>
    topClasses: Array<{ class: string; count: number }>
  }> {
    const { data } = await api.get<{
      predictions: AIPrediction[]
      hourly: Array<{ time: string; count: number }>
      topClasses: Array<{ class: string; count: number }>
    }>('/api/ai-analysis/predictive')
    return (
      data ?? {
        predictions: [],
        hourly: [],
        topClasses: [],
      }
    )
  },

  /** Automated response / playbook executions */
  async getAutomatedResponse(): Promise<{
    total: number
    success: number
    failed: number
    executions: PlaybookExecution[]
  }> {
    const { data } = await api.get<{
      total: number
      success: number
      failed: number
      executions: PlaybookExecution[]
    }>('/api/ai-analysis/automated-response')
    return (
      data ?? {
        total: 0,
        success: 0,
        failed: 0,
        executions: [],
      }
    )
  },

  /** OSINT correlation */
  async getOsint(): Promise<{
    correlated: number
    sources: string[]
    lastUpdated: string | null
  }> {
    const { data } = await api.get<{
      correlated: number
      sources: string[]
      lastUpdated: string | null
    }>('/api/ai-analysis/osint')
    return (
      data ?? {
        correlated: 0,
        sources: [],
        lastUpdated: null,
      }
    )
  },
}
