import { axiosInstance } from '../lib/axios';

interface ThreatIndicator {
  id: string;
  type: string;
  value: string;
  confidence: number;
  source: string;
  last_seen: string;
}

export const threatIntelligenceService = {
  async getIndicators(params?: {
    type?: string;
    confidence?: number;
    source?: string;
  }): Promise<ThreatIndicator[]> {
    const response = await axiosInstance.get('/threat-intel/indicators', { params });
    return response.data;
  },

  async checkIoC(value: string): Promise<{
    is_malicious: boolean;
    confidence: number;
    sources: string[];
  }> {
    const response = await axiosInstance.post('/threat-intel/check', { value });
    return response.data;
  }
};