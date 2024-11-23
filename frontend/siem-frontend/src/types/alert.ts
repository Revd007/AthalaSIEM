export interface Alert {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'new' | 'acknowledged' | 'resolved' | 'closed';
  timestamp: string;
  source: string;
  acknowledged: boolean;
}

export interface AlertResponse {
  alerts: Alert[];
  total: number;
}
