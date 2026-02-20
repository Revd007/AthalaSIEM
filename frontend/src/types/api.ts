export interface ApiResponse<T = any> {
  data: T;
  error?: string;
  message?: string;
  status: number;
  statusText: string;
  headers: Headers;
  token?: string;
  token_type?: string;
}

// --- AI / Python backend response types ---

export interface AIAnalysisOverview {
  activeThreats: number;
  avgConfidence: number;
  detectionRate24h: number;
  responseTime: string;
  mitreCoveragePercent: number;
  insightsTrend: Array<{ time: string; value: number }>;
  latestInsights: Array<{
    id: string;
    predictedClass: string;
    confidence: number;
    createdAt: string | null;
  }>;
}

export interface AIAnomaly {
  id: string;
  logId: string;
  severity: string;
  score: number;
  description?: string;
  createdAt: string;
}

export interface AIPrediction {
  id: string;
  logId: string;
  predictedClass: string;
  confidence: number;
  createdAt: string;
}

export interface DetectionRule {
  id: string;
  name: string;
  type: 'yara' | 'sigma';
  content: string;
  enabled: boolean;
  createdAt?: string;
  updatedAt?: string;
}

export interface ThreatIntelIndicator {
  id: string;
  type: string;
  value: string;
  confidence: number;
  source?: string;
  lastSeen?: string;
}

export interface PlaybookDefinition {
  id: string;
  name: string;
  description?: string;
  steps?: unknown;
  enabled: boolean;
  createdAt?: string;
  updatedAt?: string;
}

export interface PlaybookExecution {
  id: string;
  playbookId: string;
  status: string;
  startedAt: string;
  completedAt?: string;
  result?: unknown;
} 