import { api } from '@/lib/api';

interface ThreatIndicator {
  id: string;
  type: string;
  value: string;
  confidence: number;
  source: string;
  last_seen: string;
}

interface ThreatFeed {
  id: string;
  name: string;
  provider: string;
  type: 'ip' | 'domain' | 'hash' | 'url';
  lastUpdate: string;
  status: 'active' | 'disabled';
  indicators: number;
  matches: number;
}

interface ThreatIntelligenceData {
  feeds: ThreatFeed[];
  indicators: ThreatIndicator[];
  totalIndicators: number;
  totalMatches: number;
}

export const threatIntelligenceService = {
  async getThreatIntelligence(): Promise<ThreatIntelligenceData> {
    try {
      const { data } = await api.get<{
        feeds?: Array<{ name: string; indicators: number; matches: number }>;
        totalIndicators?: number;
        totalMatches?: number;
      }>('/api/threatintelligence/summary');

      if (data?.feeds && Array.isArray(data.feeds)) {
        const feeds: ThreatFeed[] = data.feeds.map((f, index) => ({
          id: `feed-${index}-${f.name}`,
          name: f.name,
          provider: 'System',
          type: 'ip' as const,
          lastUpdate: new Date().toISOString(),
          status: 'active' as const,
          indicators: f.indicators ?? 0,
          matches: f.matches ?? 0,
        }));
        return {
          feeds,
          indicators: [],
          totalIndicators: data.totalIndicators ?? feeds.reduce((s, f) => s + f.indicators, 0),
          totalMatches: data.totalMatches ?? 0,
        };
      }

      return {
        feeds: [],
        indicators: [],
        totalIndicators: 0,
        totalMatches: 0,
      };
    } catch (error) {
      console.error('Error fetching threat intelligence:', error);
      return {
        feeds: [],
        indicators: [],
        totalIndicators: 0,
        totalMatches: 0,
      };
    }
  },

  async getIndicators(params?: {
    type?: string;
    confidence?: number;
    source?: string;
  }): Promise<ThreatIndicator[]> {
    try {
      const queryString = new URLSearchParams();
      if (params?.type) queryString.append('type', params.type);
      if (params?.confidence) queryString.append('confidence', params.confidence.toString());
      if (params?.source) queryString.append('source', params.source);
      
      const { data } = await api.get<ThreatIndicator[]>(`/api/threatintelligence/indicators?${queryString.toString()}`);
      return data ?? [];
    } catch (error) {
      console.error('Error fetching indicators:', error);
      return [];
    }
  },

  async checkIoC(value: string): Promise<{
    is_malicious: boolean;
    confidence: number;
    sources: string[];
  }> {
    try {
      const { data } = await api.post('/api/threatintelligence/check', { value });
      return data ?? { is_malicious: false, confidence: 0, sources: [] };
    } catch (error) {
      console.error('Error checking IoC:', error);
      return { is_malicious: false, confidence: 0, sources: [] };
    }
  }
};