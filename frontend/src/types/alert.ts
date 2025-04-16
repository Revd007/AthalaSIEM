export interface Alert {
  id: string
  title: string
  description: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'new' | 'in_progress' | 'resolved' | 'dismissed'
  source: string
  timestamp: string
  details?: Record<string, any>
}

export interface AlertFilters {
  severity?: Alert['severity']
  status?: Alert['status']
  search?: string
} 