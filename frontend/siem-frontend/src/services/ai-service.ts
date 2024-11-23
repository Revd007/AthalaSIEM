import { axiosInstance } from '../lib/axios';

export interface ThreatAnalysis {
  is_threat: boolean;
  threat_score: number;
  indicators: string[];
  confidence: number;
}

export interface AnomalyResult {
  is_anomaly: boolean;
  anomaly_score: number;
  confidence: number;
}

export const aiService = {
  async analyzeThreat(eventData: any): Promise<ThreatAnalysis> {
    const response = await axiosInstance.post('/ai/threat-analysis', eventData);
    return response.data;
  },

  async detectAnomalies(features: any): Promise<AnomalyResult> {
    const response = await axiosInstance.post('/ai/anomaly-detection', features);
    return response.data;
  }
};