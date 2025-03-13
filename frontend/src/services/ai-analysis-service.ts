import { api } from '../lib/api';
import type { 
  AIAnalysisResult, 
  AIServiceStatus,
  KnowledgeGraphData,
  SystemMetrics 
} from '../types/ai-service';

export const aiAnalysisService = {
  // Get AI service status
  async getStatus() {
    const response = await api.get<AIServiceStatus>('/api/ai/status');
    return response.data;
  },

  // Analyze specific event
  async analyzeEvent(eventData: any) {
    const response = await api.post<AIAnalysisResult>('/api/ai/analyze/event', eventData);
    return response.data;
  },

  // Get knowledge graph
  async getKnowledgeGraph() {
    const response = await api.get<KnowledgeGraphData>('/api/ai/knowledge-graph');
    return response.data;
  },

  // Get system metrics
  async getSystemMetrics() {
    const response = await api.get<SystemMetrics>('/api/ai/system-metrics');
    return response.data;
  },

  // Analyze threats
  async analyzeThreat(eventData: any) {
    const response = await api.post('/api/ai/analyze/threat', eventData);
    return response.data;
  },

  // Detect anomalies
  async detectAnomalies(eventData: any) {
    const response = await api.post('/api/ai/analyze/anomaly', eventData);
    return response.data;
  },

  // Analyze behavior
  async analyzeBehavior(eventData: any) {
    const response = await api.post('/api/ai/analyze/behavior', eventData);
    return response.data;
  },

  // Get AI insights
  async getInsights(eventData: any) {
    const params = new URLSearchParams(eventData);
    const response = await api.get(`/api/ai/insights?${params}`);
    return response.data;
  }
}; 