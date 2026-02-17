import { api } from '@/lib/api'
import { useQuery } from '@tanstack/react-query'

// Types
export interface EventDistribution {
  name: string
  value: number
}

export interface DeviceType {
  name: string
  value: number
  type: string
}

export interface SeverityDistribution {
  name: string
  value: number
  color: string
}

export interface DeviceAnalytics {
  deviceData: DeviceType[]
  severityData: SeverityDistribution[]
}

export interface MonthlyMetric {
  month: string
  incidents: number
  resolved: number
  mttr: number
}

export interface SecurityKpi {
  title: string
  value: string
  change: string
  trend: 'up' | 'down'
}

export interface SecurityMetrics {
  monthlyData: MonthlyMetric[]
  kpis: SecurityKpi[]
}

export interface AnomalyDataPoint {
  timestamp: string
  baseline: number
  actual: number
  predicted: number
}

export interface ThreatDistribution {
  name: string
  value: number
  color: string
}

export interface AIAnalytics {
  anomalyData: AnomalyDataPoint[]
  threatDistribution: ThreatDistribution[]
}

export interface BehaviorDataPoint {
  time: string
  normalScore: number
  userScore: number
}

export interface BehavioralAnomaly {
  id: number
  user: string
  activity: string
  riskScore: number
  time: string
}

export interface BehavioralAnalytics {
  behaviorData: BehaviorDataPoint[]
  anomalies: BehavioralAnomaly[]
}

export interface PredictionDataPoint {
  time: string
  actual: number
  predicted: number
}

export interface RiskFactor {
  title: string
  description: string
  impact: 'low' | 'medium' | 'high'
  recommendation: string
}

export interface PredictiveAnalytics {
  predictions: PredictionDataPoint[]
  riskFactors: RiskFactor[]
}

/** Hourly event bucket returned by GET /api/analytics/events-over-time */
export interface EventsOverTimePoint {
  time: string
  total: number
  errors: number
  warnings: number
  normal: number
}

/** Summary counters returned by GET /api/analytics/dashboard-summary */
export interface DashboardSummary {
  totalLogs24h: number
  totalLogs1h: number
  criticalCount: number
  totalAlerts: number
  onlineAgents: number
  totalAgents: number
  eventsPerSecond: number
}

// API functions
export const analyticsService = {
  async getEventsDistribution(): Promise<EventDistribution[]> {
    const response = await api.get<EventDistribution[]>('/api/analytics/events-distribution')
    return response.data
  },

  async getSeverityDistribution(): Promise<SeverityDistribution[]> {
    const response = await api.get<SeverityDistribution[]>('/api/analytics/severity-distribution')
    return response.data
  },

  async getEventsOverTime(hours = 24): Promise<EventsOverTimePoint[]> {
    const response = await api.get<EventsOverTimePoint[]>(`/api/analytics/events-over-time?hours=${hours}`)
    return response.data
  },

  async getDashboardSummary(): Promise<DashboardSummary> {
    const response = await api.get<DashboardSummary>('/api/analytics/dashboard-summary')
    return response.data
  },

  async getDeviceAnalytics(): Promise<DeviceAnalytics> {
    const response = await api.get<DeviceAnalytics>('/api/analytics/device-analytics')
    return response.data
  },

  async getSecurityMetrics(): Promise<SecurityMetrics> {
    const response = await api.get<SecurityMetrics>('/api/analytics/security-metrics')
    return response.data
  },

  async getAIAnalytics(): Promise<AIAnalytics> {
    const response = await api.get<AIAnalytics>('/api/analytics/ai-analytics')
    return response.data
  },

  async getBehavioralAnalytics(): Promise<BehavioralAnalytics> {
    const response = await api.get<BehavioralAnalytics>('/api/analytics/behavioral-analytics')
    return response.data
  },

  async getPredictiveAnalytics(): Promise<PredictiveAnalytics> {
    const response = await api.get<PredictiveAnalytics>('/api/analytics/predictive-analytics')
    return response.data
  }
}

// React Query Hooks
export function useEventsDistribution() {
  return useQuery({
    queryKey: ['events-distribution'],
    queryFn: () => analyticsService.getEventsDistribution(),
    staleTime: 30000,
    refetchInterval: 30000,
  })
}

export function useSeverityDistribution() {
  return useQuery({
    queryKey: ['severity-distribution'],
    queryFn: () => analyticsService.getSeverityDistribution(),
    staleTime: 30000,
    refetchInterval: 30000,
  })
}

export function useEventsOverTime(hours = 24) {
  return useQuery({
    queryKey: ['events-over-time', hours],
    queryFn: () => analyticsService.getEventsOverTime(hours),
    staleTime: 15000,
    refetchInterval: 15000,
  })
}

export function useDashboardSummary() {
  return useQuery({
    queryKey: ['dashboard-summary'],
    queryFn: () => analyticsService.getDashboardSummary(),
    staleTime: 10000,
    refetchInterval: 10000,
  })
}

export function useDeviceAnalytics() {
  return useQuery({
    queryKey: ['device-analytics'],
    queryFn: () => analyticsService.getDeviceAnalytics(),
    staleTime: 60000,
  })
}

export function useSecurityMetrics() {
  return useQuery({
    queryKey: ['security-metrics'],
    queryFn: () => analyticsService.getSecurityMetrics(),
    staleTime: 60000,
  })
}

export function useAIAnalytics() {
  return useQuery({
    queryKey: ['ai-analytics'],
    queryFn: () => analyticsService.getAIAnalytics(),
    staleTime: 30000,
  })
}

export function useBehavioralAnalytics() {
  return useQuery({
    queryKey: ['behavioral-analytics'],
    queryFn: () => analyticsService.getBehavioralAnalytics(),
    staleTime: 30000,
  })
}

export function usePredictiveAnalytics() {
  return useQuery({
    queryKey: ['predictive-analytics'],
    queryFn: () => analyticsService.getPredictiveAnalytics(),
    staleTime: 30000,
  })
}
