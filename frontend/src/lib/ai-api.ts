/**
 * Typed API client for AI Analysis and Threat Hunting.
 * All requests go through .NET proxy (relative /api/...). No direct Python URL.
 */
import { api } from '@/lib/api'

// ─── Query keys (centralized for invalidation on login) ─────────────────────
export const aiKeys = {
  overview: ['ai', 'overview'] as const,
  anomalies: ['ai', 'anomalies', { period: '24h' }] as const,
  behavior: ['ai', 'behavior'] as const,
  predictive: ['ai', 'predictive'] as const,
  automatedResponse: ['ai', 'automated-response'] as const,
  osint: ['ai', 'osint'] as const,
  huntDashboard: ['threat-hunting', 'dashboard'] as const,
  huntIocScan: ['threat-hunting', 'ioc-scan'] as const,
  huntBehavior: ['threat-hunting', 'behavior'] as const,
  huntLive: (sessionId: string | null) => ['threat-hunting', 'live', sessionId] as const,
}

// ─── AI Analysis response types (match backendpy routers) ───────────────────
export interface AiOverviewResponse {
  activeThreats: number
  avgConfidence: number
  detectionRate24h: number
  responseTime: string
  mitreCoveragePercent: number
  insightsTrend: Array<{ time: string; value: number }>
  latestInsights: Array<{
    id: string
    predictedClass: string
    confidence: number
    createdAt: string | null
  }>
}

export interface AiAnomaliesResponse {
  anomalyScore: number
  detectedToday: number
  highSeverityAlerts: number
  totalLogsAnalyzed: number
  anomalyTimeline24h: Array<{ time: string; count: number }>
  detectedAnomalies: Array<{
    id: string
    logEntryId: string
    score: number
    severity: string
    createdAt: string | null
  }>
}

export interface AiBehaviorResponse {
  userActivityTimeline: Array<{ time: string; normalScore: number; userScore: number }>
  usersMonitored: number
  anomaliesToday: number
  avgRiskScore: number
  highRiskUsers: unknown[]
}

export interface AiPredictiveResponse {
  activePredictionsCount: number
  criticalAlerts: number
  totalAlerts24h: number
  highRiskPredictions: number
  predictionTimeline: Array<{ time: string; count: number }>
  activePredictions: Array<{
    id: string
    logEntryId: string
    predictedClass: string
    confidence: number
    explanation: string | null
    createdAt: string | null
  }>
}

export interface AiAutomatedResponseResponse {
  recentAutomatedActions: Array<{
    id: string
    playbookId: string
    status: string
    startedAt: string | null
    completedAt: string | null
  }>
}

export interface AiOsintResponse {
  osintPredictionCorrelation?: unknown[]
  totalPredictions?: number
}

// ─── Threat Hunting response types ─────────────────────────────────────────
export interface HuntDashboardResponse {
  huntActivityLast7Days: Array<{ date: string | null; count: number }>
  activeHunts: number
  totalFindings: number
  avgHuntDuration: number
  successRate: number
  recentFindings: Array<{
    id: string
    huntId: string
    description: string
    severity: string
    createdAt: string | null
  }>
}

export interface HuntIocScanResponse {
  matchesFound: number
  results: Array<{ type: string; value: string; sourceFeed: string; confidence: number }>
  historicalMatches: unknown[]
}

export interface HuntBehaviorResponse {
  mitreTechniqueCounts: Array<{ technique: string; count: number }>
  processBehavior: unknown[]
  networkBehavior: unknown[]
  userBehavior: unknown[]
}

export interface HuntLiveStartResponse {
  sessionId: string
  findingsCount: number
  status: string
}

export interface HuntLiveResultsResponse {
  sessionId: string
  status: string
  findingsCount: number
  findings: Array<{ id: string; logEntryId: string; description: string; severity: string }>
}

// ─── API functions ─────────────────────────────────────────────────────────
const safeGet = async <T>(url: string): Promise<T> => {
  const res = await api.get<T>(url)
  if (res.data == null) return {} as T
  return res.data
}

const safePost = async <T>(url: string, body?: object): Promise<T> => {
  const res = await api.post<T>(url, body ?? {})
  if (res.data == null) return {} as T
  return res.data
}

export const aiApi = {
  getOverview: () => safeGet<AiOverviewResponse>('/api/ai-analysis/overview'),
  getAnomalies: () => safeGet<AiAnomaliesResponse>('/api/ai-analysis/anomalies'),
  getBehavior: () => safeGet<AiBehaviorResponse>('/api/ai-analysis/behavior'),
  getPredictive: () => safeGet<AiPredictiveResponse>('/api/ai-analysis/predictive'),
  getAutomatedResponse: () => safeGet<AiAutomatedResponseResponse>('/api/ai-analysis/automated-response'),
  getOsint: () => safeGet<AiOsintResponse>('/api/ai-analysis/osint'),

  getHuntDashboard: () => safeGet<HuntDashboardResponse>('/api/threat-hunting/dashboard'),
  scanIoc: (body: { value: string; types?: string[] }) =>
    safePost<HuntIocScanResponse>('/api/threat-hunting/ioc/scan', body),
  getHuntBehavior: () => safeGet<HuntBehaviorResponse>('/api/threat-hunting/behavior'),
  liveHuntStart: (body: { query: string; timeRangeMinutes?: number }) =>
    safePost<HuntLiveStartResponse>('/api/threat-hunting/live/start', body),
  liveHuntResults: (sessionId: string) =>
    safeGet<HuntLiveResultsResponse>(`/api/threat-hunting/live/${sessionId}/results`),
}
