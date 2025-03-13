export interface LogEvent {
  id: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  message: string;
  category: string;
  sourceIp?: string;
  destinationIp?: string;
  protocol?: string;
  action?: string;
}

export interface AlertData {
  id: string;
  title: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  status: 'open' | 'investigating' | 'resolved';
  description: string;
  affectedSystems?: string[];
  assignedTo?: string;
}

export interface MetricData {
  timestamp: string;
  value: number;
}

export interface SecurityEvent {
  type: string;
  count: number;
  percentage: number;
}

export interface NetworkTraffic {
  time: string;
  inbound: number;
  outbound: number;
}