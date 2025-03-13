export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  suggestions?: string[];
}

export interface AIResponse {
  message: string;
  suggestions?: string[];
  confidence: number;
  context?: {
    relatedAlerts?: string[];
    securityContext?: string;
  };
}