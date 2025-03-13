export interface PredictionDetail {
  id: string
  type: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  probability: number
  impact: {
    financial: number
    operational: number
    reputational: number
  }
  timeline: {
    detected: string
    estimated: string
    window: string
  }
  source: {
    ip?: string
    location?: string
    actor?: string
    technique?: string
  }
  affectedSystems: {
    id: string
    name: string
    type: string
    criticality: string
    status: string
  }[]
  indicators: {
    id: string
    type: string
    value: string
    confidence: number
    firstSeen: string
    lastSeen: string
  }[]
  mitigationSteps: {
    id: string
    action: string
    priority: 'high' | 'medium' | 'low'
    status: 'pending' | 'in-progress' | 'completed'
    assignedTo?: string
    eta?: string
  }[]
  analysis: {
    summary: string
    methodology: string
    confidence: number
    falsePositiveRisk: number
    dataPoints: number
    modelVersion: string
    lastUpdated: string
  }
  relatedEvents: {
    id: string
    type: string
    timestamp: string
    description: string
  }[]
  recommendations: {
    id: string
    type: 'immediate' | 'short-term' | 'long-term'
    description: string
    impact: string
    effort: string
    status: 'proposed' | 'approved' | 'in-progress' | 'completed'
  }[]
} 