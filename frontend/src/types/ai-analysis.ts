export interface AnalysisRequest {
  data: any
  options?: {
    timeframe?: string
    severity?: string
    type?: string
  }
}

export interface AnalysisResult {
  id: string
  timestamp: string
  type: string
  severity: string
  description: string
  details: {
    [key: string]: any
  }
  recommendations: string[]
  confidence: number
} 