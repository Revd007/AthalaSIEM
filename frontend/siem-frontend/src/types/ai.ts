export interface AIInsight {
    title: string
    description: string
    severity?: 'low' | 'medium' | 'high'
    timestamp?: string
  }
  
  export interface AIAnalyticsData {
    insights: AIInsight[]
    anomalyDetection: {
      score: number
      trend: number
    }
    threatAnalysis: {
      events: Record<string, {
        severity: number
        confidence: number
      }>
    }
  }