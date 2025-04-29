import { api } from '@/lib/api';
import type { AnalysisResult, AnalysisRequest } from '@/types/ai-analysis';

interface AIServiceStatus {
  status: string;
  version: string;
  lastUpdate: string;
}

interface KnowledgeGraphData {
  nodes: Array<{
    id: string;
    type: string;
    properties: Record<string, unknown>;
  }>;
  edges: Array<{
    source: string;
    target: string;
    type: string;
    properties: Record<string, unknown>;
  }>;
}

interface SystemMetrics {
  cpu: {
    usage: number;
    temperature: number;
    cores: number;
  };
  memory: {
    total: number;
    used: number;
    free: number;
  };
  disk: {
    total: number;
    used: number;
    free: number;
  };
}

export const aiAnalysisService = {
  // Get AI service status
  async getStatus(): Promise<AIServiceStatus> {
    const response = await api.get<AIServiceStatus>('/api/ai/status');
    return response.data;
  },

  // Analyze specific event
  async analyzeEvent(eventData: unknown): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/analyze/event', eventData);
    return response.data;
  },

  // Get knowledge graph
  async getKnowledgeGraph(): Promise<KnowledgeGraphData> {
    const response = await api.get<KnowledgeGraphData>('/api/ai/knowledge-graph');
    return response.data;
  },

  // Get system metrics
  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await api.get<SystemMetrics>('/api/ai/system-metrics');
    return response.data;
  },

  // Analyze threats
  async analyzeThreat(eventData: unknown): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/analyze/threat', eventData);
    return response.data;
  },

  // Detect anomalies
  async detectAnomalies(eventData: unknown): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/analyze/anomaly', eventData);
    return response.data;
  },

  // Analyze behavior
  async analyzeBehavior(eventData: unknown): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/analyze/behavior', eventData);
    return response.data;
  },

  // Get AI insights
  async getInsights(eventData: Record<string, string>): Promise<AnalysisResult[]> {
    const params = new URLSearchParams(eventData);
    const response = await api.get<AnalysisResult[]>(`/api/ai/insights?${params}`);
    return response.data;
  },

  async analyze(data: unknown, config: { type: string; parameters: Record<string, unknown> }): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/analyze', { data, config });
    return response.data;
  },
  
  async getAnalysisHistory(): Promise<AnalysisResult[]> {
    const response = await api.get<AnalysisResult[]>('/api/ai/analysis/history');
    return response.data;
  },
  
  async getAnalysisDetails(id: string): Promise<AnalysisResult> {
    const response = await api.get<AnalysisResult>(`/api/ai/analysis/${id}`);
    return response.data;
  },

  async analyzeThreats(data: AnalysisRequest): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/analyze-threats', data);
    return response.data;
  },

  async getPredictiveAnalysis(data: AnalysisRequest): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/predictive-analysis', data);
    return response.data;
  },

  async getBehavioralAnalysis(data: AnalysisRequest): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/ai/behavioral-analysis', data);
    return response.data;
  }
}; 