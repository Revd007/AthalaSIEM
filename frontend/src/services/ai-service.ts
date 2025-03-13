import { api } from '../lib/api';

export interface AnalysisResult {
    threatAnalysis: {
        threat_score: number;
        confidence: number;
        indicators: string[];
    };
    anomalyAnalysis: {
        anomaly_score: number;
        is_anomaly: boolean;
        confidence: number;
    };
    riskAssessment: {
        risk_level: number;
        risk_score: number;
    };
}

export interface EventData {
    timestamp: string;
    source: string;
    type: string;
    data: Record<string, any>;
}

export interface AIService {
    analyzeEvent: (eventData: EventData) => Promise<AnalysisResult>;
    analyzeThreat: (eventData: EventData) => Promise<AnalysisResult['threatAnalysis']>;
    analyzeAnomaly: (eventData: EventData) => Promise<AnalysisResult['anomalyAnalysis']>;
    analyzeBehavior: (eventData: EventData) => Promise<any>;
    assessRisk: (eventData: EventData) => Promise<AnalysisResult['riskAssessment']>;
    getMetrics: () => Promise<any>;
    getStatus: () => Promise<any>;
}

export const aiService: AIService = {
    analyzeEvent: async (eventData) => {
        const response = await api.post<AnalysisResult>('/api/ai/analyze/event', eventData);
        return response.data;
    },

    analyzeThreat: async (eventData) => {
        const response = await api.post<AnalysisResult['threatAnalysis']>('/api/ai/analyze/threat', eventData);
        return response.data;
    },

    analyzeAnomaly: async (eventData) => {
        const response = await api.post<AnalysisResult['anomalyAnalysis']>('/api/ai/analyze/anomaly', eventData);
        return response.data;
    },

    analyzeBehavior: async (eventData) => {
        const response = await api.post('/api/ai/analyze/behavior', eventData);
        return response.data;
    },

    assessRisk: async (eventData) => {
        const response = await api.post<AnalysisResult['riskAssessment']>('/api/ai/assess/risk', eventData);
        return response.data;
    },

    getMetrics: async () => {
        const response = await api.get('/api/ai/metrics');
        return response.data;
    },

    getStatus: async () => {
        const response = await api.get('/api/ai/status');
        return response.data;
    }
}; 