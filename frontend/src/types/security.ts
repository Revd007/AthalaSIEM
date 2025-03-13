export interface SecurityThreat {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  impact: string;
  recommendations?: string[];
  timestamp?: string;
}

export interface AIAnalysisResult {
  riskScore: number;
  recommendations: string[];
  timeline: string[];
}

export interface SecurityAlert {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'new' | 'investigating' | 'resolved';
  timestamp: string;
  source: string;
  affectedAssets?: string[];
}